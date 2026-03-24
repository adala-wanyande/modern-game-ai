"""
MCTS BattleSnake agent with two improvements over vanilla UCB1:

  1. RAVE / AMAF (Rapid Action Value Estimation)
     Each node tracks not just its own visit/value statistics but also
     AMAF statistics: how often each move led to a good outcome anywhere
     in the subtree below. The UCB formula is replaced by the combined
     UCB+RAVE score. The RAVE weight decays as per-node visits grow,
     eventually trusting the direct statistics more (β → 0).

  2. Heuristic-biased expansion (Progressive History)
     When expanding, instead of picking an untried move at random, we rank
     candidate moves by the static heuristic score. This focuses early
     exploration on promising branches and compensates for the low iteration
     counts imposed by the 1 s time limit.

Both improvements are well-established in the literature:
  [3] Rimmel et al., "Biasing Monte-Carlo Simulations through RAVE Values"
  [4] Świechowski et al., "MCTS: a review of recent modifications"
"""

import math
import random
import time
import typing
from collections import defaultdict

from game_simulator import GameState, DIRECTIONS
from heuristic_agent import evaluate_move


# ── Hyperparameters ───────────────────────────────────────────────────────────

THINK_TIME     = 0.9
UCB_C          = 1.0      # exploration constant (slightly lower; RAVE helps)
RAVE_K         = 100      # equivalence parameter: β = √(k/(3n+k))
ROLLOUT_DEPTH  = 20


# ── RAVE Node ─────────────────────────────────────────────────────────────────

class RAVENode:
    __slots__ = ("state", "move", "parent", "children",
                 "visits", "value",
                 "rave_visits", "rave_value",
                 "untried_moves")

    def __init__(self, state: GameState, move: str = None,
                 parent: "RAVENode" = None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children: list["RAVENode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        # AMAF stats keyed by move string
        self.rave_visits: dict[str, int] = defaultdict(int)
        self.rave_value:  dict[str, float] = defaultdict(float)
        me = state.my_snake
        candidates = state.legal_moves(state.my_id) if me and me.alive else []
        # Improvement 2: rank by heuristic score at expansion time
        self.untried_moves: list[str] = _heuristic_ranked(candidates, state)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    # RAVE score for child selection
    def best_child(self, c: float) -> "RAVENode":
        log_parent = math.log(max(1, self.visits))

        def rave_score(node: "RAVENode") -> float:
            q = node.value / node.visits
            # UCB term
            ucb = c * math.sqrt(log_parent / node.visits)
            # RAVE term
            rv = self.rave_visits[node.move]
            if rv > 0:
                q_rave = self.rave_value[node.move] / rv
                beta = math.sqrt(RAVE_K / (3 * node.visits + RAVE_K))
                q = (1 - beta) * q + beta * q_rave
            return q + ucb

        return max(self.children, key=rave_score)

    def expand(self) -> "RAVENode":
        # Improvement 2: pop from front — best heuristic move first
        move = self.untried_moves.pop(0)
        moves = {s.id: random.choice(self.state.legal_moves(s.id))
                 for s in self.state.alive_snakes() if s.id != self.state.my_id}
        moves[self.state.my_id] = move
        new_state = self.state.step(moves)
        child = RAVENode(new_state, move=move, parent=self)
        self.children.append(child)
        return child


# ── Helpers ──────────────────────────────────────────────────────────────────

def _heuristic_ranked(moves: list[str], state: GameState) -> list[str]:
    """Sort moves best-first according to heuristic evaluation."""
    if not moves:
        return moves
    api = _state_to_api(state)
    scored = [(m, evaluate_move(m, api)) for m in moves]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [m for m, _ in scored]


def _state_to_api(state: GameState) -> dict:
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


# ── Rollout ───────────────────────────────────────────────────────────────────

def _rollout(state: GameState) -> tuple[float, list[str]]:
    """
    Simulate to depth; return (score, move_sequence_for_RAVE_update).
    """
    move_seq: list[str] = []
    for _ in range(ROLLOUT_DEPTH):
        if state.is_terminal():
            return 0.0, move_seq
        legal = state.legal_moves(state.my_id)
        api = _state_to_api(state)
        scores = {d: evaluate_move(d, api) for d in legal}
        my_move = max(scores, key=scores.__getitem__)
        move_seq.append(my_move)
        opp_moves = {s.id: random.choice(state.legal_moves(s.id))
                     for s in state.alive_snakes() if s.id != state.my_id}
        opp_moves[state.my_id] = my_move
        state = state.step(opp_moves)

    if state.is_terminal():
        return 0.0, move_seq
    me = state.my_snake
    if me is None:
        return 0.0, move_seq
    alive = state.alive_snakes()
    survival = me.health / 100.0
    space = state.flood_fill(me.head) / (state.width * state.height)
    length_score = me.length / max(s.length for s in alive)
    score = 0.5 * survival + 0.3 * space + 0.2 * length_score
    return score, move_seq


# ── MCTS with RAVE ────────────────────────────────────────────────────────────

def mcts_rave(root_state: GameState, think_time: float = THINK_TIME) -> str:
    root = RAVENode(root_state)
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
        result, move_seq = _rollout(node.state.copy())

        # Backpropagation with RAVE update
        move_set = set(move_seq)
        ancestor = node
        while ancestor is not None:
            ancestor.visits += 1
            ancestor.value += result
            # Update RAVE stats for every move seen in rollout
            for m in move_set:
                ancestor.rave_visits[m] += 1
                ancestor.rave_value[m] += result
            ancestor = ancestor.parent

        iterations += 1

    print(f"MCTS-RAVE: {iterations} iterations")

    if not root.children:
        legal = root_state.legal_moves(root_state.my_id)
        return random.choice(legal) if legal else "down"

    best = max(root.children, key=lambda n: n.visits)
    return best.move


# ── Battlesnake API ───────────────────────────────────────────────────────────

def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "",
        "color": "#6A1B9A",
        "head": "pixel",
        "tail": "pixel",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    state = GameState.from_api(game_state)
    best = mcts_rave(state)
    print(f"MOVE {game_state['turn']}: {best}")
    return {"move": best}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter-snake-python"))
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
