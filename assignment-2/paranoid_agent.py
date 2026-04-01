"""
Paranoid search BattleSnake agent with alpha-beta pruning.

The "paranoid" assumption treats ALL opponents as a single adversarial
coalition cooperating to eliminate us. This reduces the multi-player game
to a classic 2-player MAX vs MIN problem, enabling standard alpha-beta pruning.

Tree structure (sequential-move model within each turn):
  Each game turn is decomposed into sequential per-snake move choices:
    - Our snake's node: MAX  — we maximise our evaluation score
    - Each opponent node: MIN — opponents cooperate to minimise our score
  After all live snakes have chosen a move, the game state advances one turn
  and the process repeats.

Iterative deepening:
  We deepen the search from depth 1 upward until the time budget expires,
  always keeping the best move found at the last *fully completed* depth.

Move ordering:
  MAX nodes: moves ranked best-first by static heuristic (speeds cutoffs).
  MIN nodes: moves ranked worst-for-us first by static heuristic (speeds cutoffs).

Literature:
  [1] Sturtevant & Bowling (2006). "Effective Move Pruning in Multi-Player Games."
      AAAI Workshop on Learning for Search.
  [2] Luckhardt & Irani (1986). "An Algorithmic Solution of N-Person Games."
      AAAI-86, pp. 158–162.
  [3] Knuth & Moore (1975). "An Analysis of Alpha-Beta Pruning."
      Artificial Intelligence 6(4), pp. 293–326.
"""

import time
import typing
from copy import deepcopy

from game_simulator import GameState, DIRECTIONS
from heuristic_agent import evaluate_move


# ── Hyperparameters ───────────────────────────────────────────────────────────

THINK_TIME = 0.9   # seconds budget per move
MAX_DEPTH  = 5     # max full-turn depth for iterative deepening
INF        = float("inf")


# ── Evaluation function ───────────────────────────────────────────────────────

def _state_to_api(state: GameState) -> dict:
    """Convert GameState to API-like dict for heuristic evaluation."""
    def s2api(s):
        return {
            "id": s.id,
            "health": s.health,
            "length": s.length,
            "head": {"x": s.body[0][0], "y": s.body[0][1]},
            "body": [{"x": x, "y": y} for x, y in s.body],
        }
    me = state.my_snake or state.snakes[0]
    return {
        "turn": state.turn,
        "you": s2api(me),
        "board": {
            "width": state.width,
            "height": state.height,
            "food": [{"x": x, "y": y} for x, y in state.food],
            "hazards": [{"x": x, "y": y} for x, y in state.hazards],
            "snakes": [s2api(s) for s in state.snakes if s.alive],
        },
    }


def _evaluate(state: GameState) -> float:
    """
    Heuristic evaluation from our snake's perspective.
    Returns a value in [0, 1]; higher is better for us.
    """
    me = state.my_snake
    if me is None or not me.alive:
        return 0.0
    alive = state.alive_snakes()
    if len(alive) == 1 and alive[0].id == state.my_id:
        return 1.0   # sole survivor — maximum score
    survival   = me.health / 100.0
    space      = state.flood_fill(me.head) / (state.width * state.height)
    max_len    = max(s.length for s in alive)
    length_sc  = me.length / max_len
    return 0.5 * survival + 0.3 * space + 0.2 * length_sc


def _order_moves(moves: list[str], state: GameState, descending: bool) -> list[str]:
    """
    Sort moves by the static heuristic score.
    descending=True for MAX nodes (best move first),
    descending=False for MIN nodes (worst-for-us move first → best α cutoffs).
    """
    if len(moves) <= 1:
        return moves
    api = _state_to_api(state)
    scored = [(m, evaluate_move(m, api)) for m in moves]
    scored.sort(key=lambda t: t[1], reverse=descending)
    return [m for m, _ in scored]


# ── Recursive alpha-beta ──────────────────────────────────────────────────────

def _ab(state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        snake_order: list[str],
        snake_idx: int,
        partial_moves: dict[str, str],
        deadline: float) -> float:
    """
    Paranoid alpha-beta search (sequential-move within each turn).

    Args:
        state:         current board state (moves not yet applied for this turn)
        depth:         remaining full turns to search
        alpha, beta:   current window (MAX, MIN bounds)
        snake_order:   list of snake IDs whose turn it is, our snake first
        snake_idx:     index into snake_order of the snake choosing now
        partial_moves: moves already chosen by earlier snakes this turn
        deadline:      wall-clock time at which we must stop
    """
    # Time guard — return static eval if budget exhausted
    if time.time() > deadline:
        return _evaluate(state)

    # --- All snakes in this turn have moved: step the world ---
    if snake_idx >= len(snake_order):
        new_state = state.step(partial_moves)
        if new_state.is_terminal():
            return _evaluate(new_state)
        if depth <= 1:
            return _evaluate(new_state)
        # Rebuild snake order for next turn (only alive snakes, our snake first)
        alive_ids = {s.id for s in new_state.alive_snakes()}
        new_order = [sid for sid in snake_order if sid in alive_ids]
        if not new_order:
            return _evaluate(new_state)
        if new_state.my_id in new_order:
            new_order.remove(new_state.my_id)
            new_order.insert(0, new_state.my_id)
        return _ab(new_state, depth - 1, alpha, beta, new_order, 0, {}, deadline)

    current_id = snake_order[snake_idx]

    # Skip dead snakes
    current_snake = next((s for s in state.snakes if s.id == current_id), None)
    if current_snake is None or not current_snake.alive:
        return _ab(state, depth, alpha, beta, snake_order, snake_idx + 1,
                   partial_moves, deadline)

    is_max = (current_id == state.my_id)
    moves = state.legal_moves(current_id)
    moves = _order_moves(moves, state, descending=is_max)

    if is_max:
        # MAX node: we want to maximise
        value = -INF
        for m in moves:
            child_val = _ab(state, depth, alpha, beta, snake_order,
                            snake_idx + 1, {**partial_moves, current_id: m},
                            deadline)
            if child_val > value:
                value = child_val
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break   # β-cutoff
        return value
    else:
        # MIN node: opponents cooperate to minimise our score
        value = INF
        for m in moves:
            child_val = _ab(state, depth, alpha, beta, snake_order,
                            snake_idx + 1, {**partial_moves, current_id: m},
                            deadline)
            if child_val < value:
                value = child_val
            if value < beta:
                beta = value
            if alpha >= beta:
                break   # α-cutoff
        return value


# ── Main entry point ──────────────────────────────────────────────────────────

def paranoid(root_state: GameState, think_time: float = THINK_TIME) -> str:
    """
    Choose our snake's move using iterative-deepening paranoid alpha-beta.

    Returns the best move string found within the time budget.
    """
    deadline = time.time() + think_time
    our_moves = root_state.legal_moves(root_state.my_id)

    if not our_moves:
        return "down"
    if len(our_moves) == 1:
        return our_moves[0]

    # Build snake order: our snake first, then opponents
    alive = root_state.alive_snakes()
    snake_order = [root_state.my_id] + [
        s.id for s in alive if s.id != root_state.my_id
    ]

    # Order root moves by heuristic (best first for MAX)
    api = _state_to_api(root_state)
    root_moves = sorted(our_moves,
                        key=lambda m: evaluate_move(m, api),
                        reverse=True)

    best_move = root_moves[0]   # fallback: heuristic-best immediate move

    for depth in range(1, MAX_DEPTH + 1):
        if time.time() > deadline:
            break

        depth_best_move = root_moves[0]
        depth_best_val  = -INF
        alpha = -INF

        for move in root_moves:
            if time.time() > deadline:
                break
            # Root is a MAX node; we've already chosen 'move', continue with opponents
            partial = {root_state.my_id: move}
            val = _ab(root_state, depth, alpha, INF,
                      snake_order, 1, partial, deadline)
            if val > depth_best_val:
                depth_best_val = val
                depth_best_move = move
            alpha = max(alpha, depth_best_val)

        # Only update best_move if this depth completed without timeout
        if time.time() < deadline:
            best_move = depth_best_move
        print(f"Paranoid depth={depth}: best={depth_best_move} val={depth_best_val:.3f}")

    return best_move


# ── Battlesnake API ───────────────────────────────────────────────────────────

def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "",
        "color": "#2E7D32",
        "head": "caffeine",
        "tail": "fat-rattle",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    state = GameState.from_api(game_state)
    best = paranoid(state)
    print(f"MOVE {game_state['turn']}: {best}")
    return {"move": best}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter-snake-python"))
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
