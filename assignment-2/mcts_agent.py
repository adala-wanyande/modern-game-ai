"""
MCTS BattleSnake agent — vanilla UCB1 with heuristic rollouts.

Architecture:
  - MCTSNode wraps a GameState; children correspond to our snake's moves
  - Opponents play random legal moves during simulation
  - Rollout uses the heuristic evaluator (not random) to steer simulations
  - Budget: iterate as many times as possible within the 1 s time limit
"""

import math
import random
import time
import typing

from game_simulator import GameState, DIRECTIONS
from heuristic_agent import evaluate_move


# ── Hyperparameters ───────────────────────────────────────────────────────────

THINK_TIME    = 0.9      # seconds available per move
UCB_C         = 1.41     # exploration constant (√2)
ROLLOUT_DEPTH = 20       # max simulation depth
ROLLOUT_HEUR  = True     # use heuristic rollout (vs pure random)


# ── Node ──────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ("state", "move", "parent", "children",
                 "visits", "value", "untried_moves")

    def __init__(self, state: GameState, move: str = None,
                 parent: "MCTSNode" = None):
        self.state = state
        self.move = move          # move that led to this node
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        me = state.my_snake
        self.untried_moves: list[str] = (
            state.legal_moves(state.my_id) if me and me.alive else []
        )

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child(self, c: float) -> "MCTSNode":
        log_parent = math.log(self.visits)
        return max(
            self.children,
            key=lambda n: n.value / n.visits + c * math.sqrt(log_parent / n.visits),
        )

    def expand(self) -> "MCTSNode":
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        # All other snakes move randomly
        moves = {s.id: random.choice(self.state.legal_moves(s.id))
                 for s in self.state.alive_snakes() if s.id != self.state.my_id}
        moves[self.state.my_id] = move
        new_state = self.state.step(moves)
        child = MCTSNode(new_state, move=move, parent=self)
        self.children.append(child)
        return child


# ── MCTS ──────────────────────────────────────────────────────────────────────

def _rollout_result(state: GameState) -> float:
    """
    Simulate from state up to ROLLOUT_DEPTH steps.
    Returns a score in [0, 1] from our snake's perspective.
    """
    for _ in range(ROLLOUT_DEPTH):
        if state.is_terminal():
            return 0.0

        me = state.my_snake
        if ROLLOUT_HEUR:
            # Pick the heuristically best safe move
            legal = state.legal_moves(state.my_id)
            api_state = _state_to_api(state)
            scores = {d: evaluate_move(d, api_state) for d in legal}
            my_move = max(scores, key=scores.__getitem__)
        else:
            legal = state.legal_moves(state.my_id)
            my_move = random.choice(legal)

        moves = {s.id: random.choice(state.legal_moves(s.id))
                 for s in state.alive_snakes() if s.id != state.my_id}
        moves[state.my_id] = my_move
        state = state.step(moves)

    if state.is_terminal():
        return 0.0
    me = state.my_snake
    if me is None:
        return 0.0
    alive = state.alive_snakes()
    n_alive = len(alive)
    n_total = len(state.snakes)
    survival = me.health / 100.0
    space = state.flood_fill(me.head) / (state.width * state.height)
    length_score = me.length / max(s.length for s in alive)
    return 0.5 * survival + 0.3 * space + 0.2 * length_score


def _state_to_api(state: GameState) -> dict:
    """Convert simulator GameState back to API-like dict for heuristic eval."""
    def snake_to_api(s):
        return {
            "id": s.id,
            "health": s.health,
            "length": s.length,
            "body": [{"x": x, "y": y} for x, y in s.body],
        }

    me = state.my_snake or state.snakes[0]
    return {
        "turn": state.turn,
        "you": snake_to_api(me),
        "board": {
            "width": state.width,
            "height": state.height,
            "food": [{"x": x, "y": y} for x, y in state.food],
            "hazards": [{"x": x, "y": y} for x, y in state.hazards],
            "snakes": [snake_to_api(s) for s in state.snakes if s.alive],
        },
    }


def mcts(root_state: GameState, think_time: float = THINK_TIME) -> str:
    root = MCTSNode(root_state)
    deadline = time.time() + think_time
    iterations = 0

    while time.time() < deadline:
        # Selection
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(UCB_C)

        # Expansion
        if not node.is_fully_expanded() and not node.state.is_terminal():
            node = node.expand()

        # Simulation
        result = _rollout_result(node.state.copy())

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

        iterations += 1

    print(f"MCTS: {iterations} iterations")

    if not root.children:
        legal = root_state.legal_moves(root_state.my_id)
        return random.choice(legal) if legal else "down"

    # Pick most-visited child (robust to outliers)
    best = max(root.children, key=lambda n: n.visits)
    return best.move


# ── Battlesnake API ───────────────────────────────────────────────────────────

def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "",
        "color": "#1565C0",
        "head": "smart-caterpillar",
        "tail": "block-bum",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    state = GameState.from_api(game_state)
    best = mcts(state)
    print(f"MOVE {game_state['turn']}: {best}")
    return {"move": best}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter-snake-python"))
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
