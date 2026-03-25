import typing
import argparse
from main_MCTS import mcts_agent
from main_heuristic import start, end 


def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Team 13",
        "color": getattr(info, "snake_color", "#CA8F2A"),
        "name": getattr(info, "snake_name", "MCTS-Snake"),
        "head": "shades",
        "tail": "bolt",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", type=str, default="MCTS-Snake")
    parser.add_argument("--color", type=str, default="#CA8F2A")
    parser.add_argument("--competitive", action="store_true")
    
    args = parser.parse_args()
    info.snake_color = args.color
    info.snake_name = args.name
    is_comp = args.competitive
    
    from server import run_server
    run_server({
        "info": info, 
        "start": start, 
        "move": lambda gs: mcts_agent(gs, True, is_comp), 
        "end": end
    }, port=args.port)


    