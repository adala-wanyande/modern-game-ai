# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    next_positions = {
        "up":    {"x": my_head["x"],     "y": my_head["y"] + 1},
        "down":  {"x": my_head["x"],     "y": my_head["y"] - 1},
        "left":  {"x": my_head["x"] - 1, "y": my_head["y"]},
        "right": {"x": my_head["x"] + 1, "y": my_head["y"]},
    }

    for direction, pos in next_positions.items():
        if pos["x"] < 0 or pos["x"] >= board_width or pos["y"] < 0 or pos["y"] >= board_height:
            is_move_safe[direction] = False

    # Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    body_set = {(seg["x"], seg["y"]) for seg in my_body[:-1]}  # tail will move away

    for direction, pos in next_positions.items():
        if (pos["x"], pos["y"]) in body_set:
            is_move_safe[direction] = False

    # Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    for snake in opponents:
        for seg in snake["body"][:-1]:
            for direction, pos in next_positions.items():
                if (pos["x"], pos["y"]) == (seg["x"], seg["y"]):
                    is_move_safe[direction] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Default: random safe move
    next_move = random.choice(safe_moves)

    # Step 4 - Move towards closest food to regain health and grow
    food = game_state['board']['food']
    if food and safe_moves:
        best_move = None
        best_dist = float("inf")
        for direction in safe_moves:
            pos = next_positions[direction]
            min_food_dist = min(
                abs(pos["x"] - f["x"]) + abs(pos["y"] - f["y"]) for f in food
            )
            if min_food_dist < best_dist:
                best_dist = min_food_dist
                best_move = direction
        next_move = best_move

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
