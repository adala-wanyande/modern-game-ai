import sys
import random
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdpc import Editor, Block, geometry

# =============================================================================
# 1. SETUP
# =============================================================================
editor = Editor(buffering=True)

try:
    buildArea = editor.getBuildArea()
    buildRect = buildArea.toRect()
    print(f"Build area: {buildRect}")
except Exception as e:
    print(f"Error getting build area: {e}")
    print("Run  /buildarea set ~ ~ ~ ~100 ~ ~100  in Minecraft first!")
    sys.exit(1)

ORIGIN_X = int(buildRect.begin.x)
ORIGIN_Z = int(buildRect.begin.y)
SPAN_X   = int(buildRect.size.x)
SPAN_Z   = int(buildRect.size.y)

# =============================================================================
# 2. LOAD WORLD SLICE
# =============================================================================
print("Loading world slice...")
worldSlice = editor.loadWorldSlice(buildRect)

hmap_surface = np.array(worldSlice.heightmaps["MOTION_BLOCKING_NO_LEAVES"], dtype=np.int32)
hmap_ocean   = np.array(worldSlice.heightmaps["OCEAN_FLOOR"], dtype=np.int32)

def get_surface_height(x: int, z: int) -> int:
    rx, rz = x - ORIGIN_X, z - ORIGIN_Z
    if 0 <= rx < SPAN_X and 0 <= rz < SPAN_Z:
        return int(hmap_surface[rx, rz])
    return 64

def get_solid_ground_height(x: int, z: int) -> int:
    rx, rz = x - ORIGIN_X, z - ORIGIN_Z
    if 0 <= rx < SPAN_X and 0 <= rz < SPAN_Z:
        return int(hmap_ocean[rx, rz])
    return 60

# =============================================================================
# 3. TERRAIN SUITABILITY
# =============================================================================
SEA_LEVEL = 62
SCORE_W   = 26  
SCORE_D   = 26

print("Computing terrain suitability...")
patches = sliding_window_view(hmap_surface.astype(np.float64), (SCORE_W, SCORE_D))
vx, vz = patches.shape[0], patches.shape[1]

win_std  = patches.std(axis=(-2, -1))
roughness = np.full((SPAN_X, SPAN_Z), np.inf, dtype=np.float64)
roughness[:vx, :vz] = win_std
suit_map = -roughness.copy()

def save_terrain_plots():
    axes = plt.subplots(1, 3, figsize=(16, 5))[1]
    im0 = axes[0].imshow(hmap_surface.T, origin="lower", cmap="terrain")
    axes[0].set_title("Heightmap")
    plt.colorbar(im0, ax=axes[0])
    
    valid = suit_map[np.isfinite(suit_map)]
    if valid.size > 0:
        norm = (suit_map - valid.min()) / (valid.max() - valid.min())
        norm[~np.isfinite(suit_map)] = np.nan
        im2 = axes[2].imshow(norm.T, origin="lower", cmap="RdYlGn")
    else:
        im2 = axes[2].imshow(suit_map.T, origin="lower", cmap="gray")
    axes[2].set_title("Suitability")
    plt.colorbar(im2, ax=axes[2])
    plt.savefig("terrain_analysis.png")
    plt.close()

save_terrain_plots()

# =============================================================================
# 4. SITE SELECTION
# =============================================================================
valid_mask = np.isfinite(suit_map)

if valid_mask.any():
    threshold = np.percentile(suit_map[valid_mask], 75)
    candidates = list(zip(*np.where(suit_map >= threshold)))
    rx0, rz0 = random.choice(candidates)
    start_x = ORIGIN_X + int(rx0) + 4
    start_z = ORIGIN_Z + int(rz0) + 4
    print(f"Selected estate site at {start_x}, {start_z}")
else:
    start_x = ORIGIN_X + SPAN_X // 2
    start_z = ORIGIN_Z + SPAN_Z // 2

# =============================================================================
# 5. GRAND MANOR & ESTATE GENERATOR
# =============================================================================

def build_estate(x, z):
    width = 11  
    depth = 15  
    
    # Estate Boundaries
    garden_x1 = x - 6
    garden_x2 = x + width + 6
    garden_z1 = z - 6
    garden_z2 = z + depth + 6

    center_y = get_solid_ground_height(x + width//2, z + depth//2)
    y = max(center_y, SEA_LEVEL + 1)
    
    print(f"Terraforming Estate Platform at {x}, {y}, {z}")

    # --- 1. ESTATE TERRAFORMING & TREE CLEARING BUFFER ---
    CLEAR_BUFFER = 7
    
    # A) Clear the sky above the estate and buffer (removes mountains/canopies)
    geometry.placeCuboid(
        editor, 
        (garden_x1 - CLEAR_BUFFER, y, garden_z1 - CLEAR_BUFFER), 
        (garden_x2 + CLEAR_BUFFER, y + 30, garden_z2 + CLEAR_BUFFER), 
        Block("air")
    )

    # B) The "Perimeter Scrubber" - Removes trees/leaves below 'y' outside the walls
    for i in range(garden_x1 - CLEAR_BUFFER, garden_x2 + CLEAR_BUFFER + 1):
        for j in range(garden_z1 - CLEAR_BUFFER, garden_z2 + CLEAR_BUFFER + 1):
            is_inside_estate = (garden_x1 <= i <= garden_x2) and (garden_z1 <= j <= garden_z2)
            if not is_inside_estate:
                ground_y = get_surface_height(i, j)
                # If ground is below estate level, clear everything from ground up to the estate level
                if ground_y < y:
                    geometry.placeCuboid(editor, (i, ground_y + 1, j), (i, y, j), Block("air"))

    # C) Build the engineered solid ground layer inside the estate
    for i in range(garden_x1, garden_x2 + 1):
        for j in range(garden_z1, garden_z2 + 1):
            is_house = (x <= i < x + width) and (z <= j < z + depth)
            is_border = (i == garden_x1) or (i == garden_x2) or (j == garden_z1) or (j == garden_z2)
            
            solid_y = get_solid_ground_height(i, j)
            
            if solid_y < y - 1:
                # Solid stone for the house base and walls, solid dirt for the garden
                if is_border or is_house:
                    fill_mat = "mossy_stone_bricks" if solid_y < SEA_LEVEL else "stone_bricks"
                else:
                    fill_mat = "dirt"
                # This completely overwrites/crushes anything below the estate!
                geometry.placeCuboid(editor, (i, solid_y, j), (i, y - 2, j), Block(fill_mat))
            
            # Cap the surface
            if is_house or is_border:
                editor.placeBlock((i, y - 1, j), Block("stone_bricks"))
            else:
                editor.placeBlock((i, y - 1, j), Block("grass_block"))

    # --- 2. ESTATE PERIMETER WALL ---
    for i in range(garden_x1, garden_x2 + 1):
        for j in range(garden_z1, garden_z2 + 1):
            if (i == garden_x1) or (i == garden_x2) or (j == garden_z1) or (j == garden_z2):
                editor.placeBlock((i, y, j), Block("cobblestone_wall"))
                if (i % 6 == 0 and j in (garden_z1, garden_z2)) or (j % 6 == 0 and i in (garden_x1, garden_x2)):
                    editor.placeBlock((i, y+1, j), Block("lantern"))

    # --- 3. GROUND FLOOR (Stone Base) ---
    geometry.placeCuboid(editor, (x, y, z), (x+width-1, y, z+depth-1), Block("spruce_planks"))
    geometry.placeCuboid(editor, (x, y+1, z), (x+width-1, y+4, z+depth-1), Block("stone_bricks"))
    geometry.placeCuboid(editor, (x+1, y+1, z+1), (x+width-2, y+4, z+depth-2), Block("air"))
    geometry.placeCuboid(editor, (x-1, y, z-1), (x+width, y, z+depth), Block("stone_brick_stairs", {"half": "top"}))

    # --- 4. SECOND FLOOR (Tudor Timber Overhang) ---
    y2 = y + 5
    geometry.placeCuboid(editor, (x-1, y2, z-1), (x+width, y2, z+depth), Block("spruce_planks"))
    geometry.placeCuboid(editor, (x-1, y2+1, z-1), (x+width, y2+4, z+depth), Block("white_terracotta"))
    geometry.placeCuboid(editor, (x, y2+1, z), (x+width-1, y2+4, z+depth-1), Block("air"))
    
    for i in range(x-1, x+width+1, 3):
        if i <= x+width:
            geometry.placeCuboid(editor, (i, y2+1, z-1), (i, y2+4, z-1), Block("stripped_dark_oak_log"))
            geometry.placeCuboid(editor, (i, y2+1, z+depth), (i, y2+4, z+depth), Block("stripped_dark_oak_log"))
    for j in range(z-1, z+depth+1, 3):
        if j <= z+depth:
            geometry.placeCuboid(editor, (x-1, y2+1, j), (x-1, y2+4, j), Block("stripped_dark_oak_log"))
            geometry.placeCuboid(editor, (x+width, y2+1, j), (x+width, y2+4, j), Block("stripped_dark_oak_log"))

    # --- 5. A-FRAME ROOF ---
    roof_y = y2 + 5
    mid_x = x + width // 2
    for i in range((width+2) // 2 + 1):
        curr_y = roof_y + i
        geometry.placeCuboid(editor, (x-1+i, curr_y, z-1), (x+width-i, curr_y, z-1), Block("white_terracotta"))
        geometry.placeCuboid(editor, (x-1+i, curr_y, z+depth), (x+width-i, curr_y, z+depth), Block("white_terracotta"))

    for i in range((width+2) // 2 + 1):
        curr_y = roof_y + i
        left_x = x - 2 + i
        right_x = x + width + 1 - i
        geometry.placeCuboid(editor, (left_x, curr_y, z-2), (left_x, curr_y, z+depth+1), Block("dark_oak_stairs", {"facing": "east"}))
        geometry.placeCuboid(editor, (right_x, curr_y, z-2), (right_x, curr_y, z+depth+1), Block("dark_oak_stairs", {"facing": "west"}))
        editor.placeBlock((left_x, curr_y, z-2), Block("deepslate_tile_stairs", {"facing": "east"}))
        editor.placeBlock((left_x, curr_y, z+depth+1), Block("deepslate_tile_stairs", {"facing": "east"}))
        editor.placeBlock((right_x, curr_y, z-2), Block("deepslate_tile_stairs", {"facing": "west"}))
        editor.placeBlock((right_x, curr_y, z+depth+1), Block("deepslate_tile_stairs", {"facing": "west"}))

    ridge_y = roof_y + ((width+2)//2)
    geometry.placeCuboid(editor, (mid_x, ridge_y, z-2), (mid_x, ridge_y, z+depth+1), Block("deepslate_tile_slab"))

    # --- 6. CHIMNEY & EXTERIOR DETAILS ---
    cx, cz = x + 1, z + depth // 2
    geometry.placeCuboid(editor, (cx, y+1, cz-1), (cx+1, y+3, cz+1), Block("bricks"))
    editor.placeBlock((cx, y+1, cz), Block("campfire"))
    editor.placeBlock((cx+1, y+1, cz), Block("iron_bars"))
    geometry.placeCuboid(editor, (cx, y+4, cz), (cx, roof_y + 6, cz), Block("bricks"))
    editor.placeBlock((cx, roof_y + 8, cz), Block("campfire")) 

    door_x = x + width // 2
    editor.placeBlock((door_x, y+1, z), Block("dark_oak_door", {"half":"lower", "facing":"north"}))
    editor.placeBlock((door_x, y+2, z), Block("dark_oak_door", {"half":"upper", "facing":"north"}))
    
    editor.placeBlock((door_x, y, z-1), Block("stone_brick_stairs", {"facing": "north"}))

    editor.placeBlock((x+2, y+2, z), Block("glass_pane"))
    editor.placeBlock((x+width-3, y+2, z), Block("glass_pane"))
    editor.placeBlock((door_x, y2+2, z-1), Block("glass_pane"))
    editor.placeBlock((door_x, y2+1, z-2), Block("dirt"))
    editor.placeBlock((door_x, y2+2, z-2), Block("cornflower"))
    editor.placeBlock((door_x-1, y2+1, z-2), Block("spruce_trapdoor", {"facing": "north", "open": "true"}))
    editor.placeBlock((door_x+1, y2+1, z-2), Block("spruce_trapdoor", {"facing": "north", "open": "true"}))

    # --- 7. GARDEN LANDSCAPING ---
    for pz in range(garden_z1 + 1, z):
        editor.placeBlock((door_x, y-1, pz), Block("gravel"))
        editor.placeBlock((door_x-1, y-1, pz), Block("gravel"))
        editor.placeBlock((door_x+1, y-1, pz), Block("gravel"))

    fx, fz = x + width + 3, z + 4
    geometry.placeCuboid(editor, (fx-2, y, fz-2), (fx+2, y, fz+2), Block("stone_bricks"))
    geometry.placeCuboid(editor, (fx-1, y, fz-1), (fx+1, y, fz+1), Block("water"))
    editor.placeBlock((fx, y+1, fz), Block("stone_brick_wall"))
    editor.placeBlock((fx, y+2, fz), Block("stone_brick_wall"))
    editor.placeBlock((fx, y+3, fz), Block("water"))

    for _ in range(18):
        tx = random.randint(garden_x1 + 1, garden_x2 - 1)
        tz = random.randint(garden_z1 + 1, garden_z2 - 1)
        if not (x-1 <= tx <= x+width and z-1 <= tz <= z+depth) and tx not in (door_x, door_x-1, door_x+1):
            if random.random() > 0.5:
                editor.placeBlock((tx, y, tz), Block("oak_fence"))
                editor.placeBlock((tx, y+1, tz), Block("oak_fence"))
                geometry.placeCuboid(editor, (tx-1, y+2, tz-1), (tx+1, y+3, tz+1), Block("oak_leaves"))
            else:
                flora = random.choice(["oak_leaves", "rose_bush", "peony", "lilac"])
                editor.placeBlock((tx, y, tz), Block(flora))

# =============================================================================
# 6. EXECUTION
# =============================================================================
if __name__ == "__main__":
    try:
        build_estate(start_x, start_z)
        editor.flushBuffer()
        print("Done! Grand Manor and Terraformed Estate built successfully.")
    except Exception as e:
        print(f"An error occurred during generation: {e}")