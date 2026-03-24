"""
Heuristic BattleSnake agent.

State evaluation uses:
  - Flood-fill reachable area (open space)
  - Closest food distance (weighted by health urgency)
  - Hazard tile penalty
  - Head-to-head risk vs larger/equal snakes
"""

import random
import typing
from collections import deque


# ── Helpers ──────────────────────────────────────────────────────────────────

def _next_pos(pos: dict, direction: str) -> dict:
    if direction == "up":
        return {"x": pos["x"], "y": pos["y"] + 1}
    if direction == "down":
        return {"x": pos["x"], "y": pos["y"] - 1}
    if direction == "left":
        return {"x": pos["x"] - 1, "y": pos["y"]}
    return {"x": pos["x"] + 1, "y": pos["y"]}


def _next_positions(head: dict) -> dict[str, dict]:
    return {d: _next_pos(head, d) for d in ("up", "down", "left", "right")}


def _safe_moves(head: dict, board_w: int, board_h: int,
                blocked: set[tuple]) -> list[str]:
    safe = []
    for d, pos in _next_positions(head).items():
        x, y = pos["x"], pos["y"]
        if 0 <= x < board_w and 0 <= y < board_h and (x, y) not in blocked:
            safe.append(d)
    return safe


def _blocked_cells(game_state: dict) -> set[tuple]:
    """All snake body segments (excluding tails, which will vacate)."""
    blocked = set()
    for snake in game_state["board"]["snakes"]:
        for seg in snake["body"][:-1]:
            blocked.add((seg["x"], seg["y"]))
    return blocked


def _flood_fill(start: dict, board_w: int, board_h: int,
                blocked: set[tuple]) -> int:
    """BFS flood fill – returns number of reachable cells from start."""
    sx, sy = start["x"], start["y"]
    if (sx, sy) in blocked:
        return 0
    visited = {(sx, sy)}
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < board_w and 0 <= ny < board_h
                    and (nx, ny) not in visited
                    and (nx, ny) not in blocked):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return len(visited)


def _manhattan(a: dict, b: dict) -> int:
    return abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])


# ── State evaluator ───────────────────────────────────────────────────────────

def evaluate_move(direction: str, game_state: dict) -> float:
    """Return a scalar score for moving in *direction* from current state."""
    board = game_state["board"]
    board_w, board_h = board["width"], board["height"]
    me = game_state["you"]
    my_head = me["body"][0]
    my_len = me["length"]
    my_health = me["health"]

    next_head = _next_pos(my_head, direction)
    nx, ny = next_head["x"], next_head["y"]

    # 1. Out-of-bounds → -inf
    if nx < 0 or nx >= board_w or ny < 0 or ny >= board_h:
        return float("-inf")

    blocked = _blocked_cells(game_state)

    # 2. Body collision → -inf
    if (nx, ny) in blocked:
        return float("-inf")

    score = 0.0

    # 3. Flood fill – open space (most important survival signal)
    # Temporarily mark next_head as visited in blocked for fill from next_head
    space = _flood_fill(next_head, board_w, board_h, blocked)
    total_cells = board_w * board_h
    score += 40.0 * (space / total_cells)

    # 4. Food proximity (urgency scales with low health)
    food = board["food"]
    if food:
        closest_food_dist = min(_manhattan(next_head, f) for f in food)
        health_factor = max(0.0, (100 - my_health) / 100.0)  # 0..1
        urgency = 0.5 + health_factor * 1.5                   # 0.5..2.0
        score += urgency * max(0, 10 - closest_food_dist)

    # 5. Hazard penalty (hazards only hurt on head-on collision; body safe)
    hazards = {(h["x"], h["y"]) for h in board.get("hazards", [])}
    if (nx, ny) in hazards:
        score -= 15.0

    # 6. Head-to-head risk: penalise if an equal/larger snake head could land here
    for snake in board["snakes"]:
        if snake["id"] == me["id"]:
            continue
        their_head = snake["body"][0]
        their_len = snake["length"]
        dist = _manhattan(their_head, next_head)
        if dist <= 2:
            # They could reach our next_head in one step
            if their_len >= my_len:
                score -= 30.0  # very bad
            else:
                score += 5.0   # we can eliminate them

    return score


# ── Battlesnake API ───────────────────────────────────────────────────────────

def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "",
        "color": "#E53935",
        "head": "evil",
        "tail": "sharp",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    scores = {d: evaluate_move(d, game_state)
              for d in ("up", "down", "left", "right")}

    best = max(scores, key=scores.__getitem__)

    # If all moves are -inf, default to down
    if scores[best] == float("-inf"):
        best = "down"

    print(f"MOVE {game_state['turn']}: {best}  scores={scores}")
    return {"move": best}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter-snake-python"))
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
