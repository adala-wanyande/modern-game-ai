
import random
import typing
from collections import deque
import argparse

move_counter = 0
def manhattan_dist(a, b):
        return abs(a['x'] - b['x']) + abs(a['y'] - b['y'])

def count_open_space(start_pos, body_coords, opponent_coords, hazard_coords, board_width, board_height, max_depth=5):
    """
    Breadth-first search to count the number of reachable open tiles from a starting position.
    Avoids the snake's own body, opponent bodies, and hazards. Used to estimate how much space is available for a move.

    Inputs:
    -start_pos (dict): Starting position {"x": int, "y": int}
    - body_coords (list): List of dicts for your snake's body segments
    - opponent_coords (set): Set of (x, y) tuples for all opponent body segments
    - hazard_coords (set): Set of (x, y) tuples for all hazard tiles
    - board_width (int): Board width
    - board_height (int): Board height
    - max_depth (int): Maximum search depth (default 5)

    Returns:
    - int: Number of reachable open tiles (including start_pos)
    """
    visited = set()
    queue = deque()
    queue.append((start_pos['x'], start_pos['y'], 0))
    while queue:
        x, y, depth = queue.popleft()
        if (x, y) in visited or depth > max_depth:
            continue
        visited.add((x, y))
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < board_width and 0 <= ny < board_height:
                if {'x': nx, 'y': ny} in body_coords:
                    continue
                if (nx, ny) in opponent_coords:
                    continue
                if (nx, ny) in hazard_coords:
                    continue
                queue.append((nx, ny, depth+1))
    return len(visited)

def heuristic_agent(game_state: typing.Dict, competitive: bool = False, mcts_use = False) -> typing.Dict:
    """
    A heuristic-based Battlesnake agent that evaluates moves based on safety, proximity to food, and open space. Movers are based on a series of rules to avoid hazards, prevent collisions, and target food effectively.
    In competitive mode we prioritize food that is farther from opponents, while in peaceful mode we target the closest food. The agent also considers open space to avoid getting trapped.
    Inputs:
    - game_state (dict): The current game state provided by the Battlesnake engine.
    - competitive (bool): Whether to play competitively (default False).
    - mcts_use (bool): Whether to use MCTS for move selection (default False).
    Returns:
    - dict: The chosen move direction."""
    global move_counter
    move_counter += 1
    # Step 1: Avoid moving backwards
    hazards = game_state['board'].get('hazards', [])
    hazard_coords = set((hazard['x'], hazard['y']) for hazard in hazards)
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    my_head = game_state["you"]["body"][0]
    my_neck = game_state["you"]["body"][1]
    if my_neck["x"] < my_head["x"]:
        is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]:
        is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]:
        is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]:
        is_move_safe["up"] = False

    # Step 2: Prevent moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    if my_head['x'] == 0:
        is_move_safe['left'] = False
    if my_head['x'] == board_width - 1:
        is_move_safe['right'] = False
    if my_head['y'] == 0:
        is_move_safe['down'] = False
    if my_head['y'] == board_height - 1:
        is_move_safe['up'] = False

    # Step 3: Prevent self-collision
    my_body = game_state['you']['body']
    next_positions = {
        "up":    {"x": my_head["x"],     "y": my_head["y"] + 1},
        "down":  {"x": my_head["x"],     "y": my_head["y"] - 1},
        "left":  {"x": my_head["x"] - 1, "y": my_head["y"]},
        "right": {"x": my_head["x"] + 1, "y": my_head["y"]},
    }
    body_coords = [{"x": segment["x"], "y": segment["y"]} for segment in my_body]
    for move_dir, pos in next_positions.items():
        if pos in body_coords:
            is_move_safe[move_dir] = False

    # Step 4: Prevent collision with other snakes
    opponents = game_state['board']['snakes']
    opponent_coords = set()
    for snake in opponents:
        for segment in snake['body']:
            opponent_coords.add((segment['x'], segment['y']))
    for move_dir, pos in next_positions.items():
        if (pos['x'], pos['y']) in opponent_coords:
            is_move_safe[move_dir] = False

    # Step 5: Avoid head-to-head collisions (competitive: only if not longer)
    my_length = len(game_state['you']['body'])
    opponent_heads = [snake['body'][0] for snake in opponents if snake['id'] != game_state['you']['id']]
    opponent_lengths = {snake['id']: len(snake['body']) for snake in opponents if snake['id'] != game_state['you']['id']}
    for move_dir, pos in next_positions.items():
        for opp_head in opponent_heads:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                opp_next = {'x': opp_head['x'] + dx, 'y': opp_head['y'] + dy}
                if 0 <= opp_next['x'] < board_width and 0 <= opp_next['y'] < board_height:
                    if opp_next == pos:
                        opp_snake = [snake for snake in opponents if snake['body'][0] == opp_head]
                        if competitive:
                            if opp_snake and opponent_lengths[opp_snake[0]['id']] >= my_length:
                                is_move_safe[move_dir] = False
                        else:
                            is_move_safe[move_dir] = False
    # Step 6: Build list of safe and hazard moves
    safe_moves = []
    hazard_moves = []
    for move_dir, isSafe in is_move_safe.items():
        if isSafe:
            pos = next_positions[move_dir]
            if (pos['x'], pos['y']) in hazard_coords:
                hazard_moves.append(move_dir)
            else:
                safe_moves.append(move_dir)
    # Prefer safe moves, but use hazard moves if no other options
    candidate_moves = safe_moves if safe_moves else hazard_moves

    # Step 7: If no moves are possible, default to down
    if not candidate_moves:
        if mcts_use == False:
            print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Step 8: Food targeting logic
    food = game_state['board']['food']
    closest_food = None
    if competitive:
        # In competitive mode, target food farthest from opponents
        best_food = None
        max_min_opp_dist = -1
        for f in food:
            my_dist = manhattan_dist(my_head, f)
            if opponent_heads:
                min_opp_dist = min(manhattan_dist(opp_head, f) for opp_head in opponent_heads)
            else:
                min_opp_dist = 0
            if min_opp_dist > max_min_opp_dist or (min_opp_dist == max_min_opp_dist and (best_food is None or my_dist < manhattan_dist(my_head, best_food))):
                max_min_opp_dist = min_opp_dist
                best_food = f
        closest_food = best_food
    else:
        # In peaceful mode, target closest food
        min_dist = float('inf')
        for f in food:
            dist = manhattan_dist(my_head, f)
            if dist < min_dist:
                min_dist = dist
                closest_food = f

    # Step 9: Score moves by open space and food distance
    move_scores = []
    for move_dir in candidate_moves:
        pos = next_positions[move_dir]
        temp_body = body_coords[:-1] if pos not in food else body_coords
        open_space = count_open_space(pos, temp_body, opponent_coords, hazard_coords, board_width, board_height)
        food_dist = manhattan_dist(pos, closest_food) if closest_food else 0
        move_scores.append((open_space, -food_dist, move_dir, food_dist))

    # Step 10: Choose best move based on health and scoring
    health = game_state['you']['health']
    if (competitive and health <= 60 and closest_food) or (not competitive and health <= 40 and closest_food):
        # When low on health, prioritize food
        min_food_dist = min(s[3] for s in move_scores)
        food_moves = [s for s in move_scores if s[3] == min_food_dist]
        max_open = max(s[0] for s in food_moves)
        best_moves = [s[2] for s in food_moves if s[0] == max_open]
        next_move = random.choice(best_moves)
    else:
        # Otherwise, prefer open space, then food
        if move_scores:
            max_open = max(s[0] for s in move_scores)
            open_moves = [s for s in move_scores if s[0] == max_open]
            best_score = max(s[1] for s in open_moves)
            best_moves = [s[2] for s in open_moves if s[1] == best_score]
            next_move = random.choice(best_moves)
        else:
            next_move = random.choice(candidate_moves)

    # Step 11: Return the chosen move
    if mcts_use == False:
        print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Team 13",
        "color": getattr(info, "snake_color", "#5DD165"),
        "name": getattr(info, "snake_name", "Heuristic-Snake"),
        "head": "default",
        "tail": "default",
    }

def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", type=str, default="Heuristic-Snake")
    parser.add_argument("--color", type=str, default="#5DD165")
    parser.add_argument("--competitive", action="store_true")
    args = parser.parse_args()

    info.snake_color = args.color
    info.snake_name = args.name
    is_comp = args.competitive

    from server import run_server
    run_server({"info": info, "start": start, "move": lambda gs: heuristic_agent(gs, is_comp), "end": end}, port=args.port)