"""
Local BattleSnake tournament runner with TrueSkill scoring.

Usage:
    python tournament.py [--games N] [--seed S]

The script:
  1. Runs N simulated games using the game_simulator (no HTTP servers needed)
  2. Assigns each snake slot to an agent strategy
  3. Records outcomes and computes TrueSkill ratings after every game
  4. Prints a final leaderboard

Agent strategies available:
  - random      : random legal moves
  - heuristic   : flood-fill + food heuristic (heuristic_agent.py)
  - mcts        : vanilla MCTS (mcts_agent.py)
  - mcts_rave   : MCTS + RAVE improvements (mcts_rave_agent.py)
"""

import argparse
import random
import sys
import time
from copy import deepcopy

import trueskill

from game_simulator import GameState, Snake, DIRECTIONS


# ── Agent interfaces ──────────────────────────────────────────────────────────

def _state_to_api(state: GameState, snake_id: str) -> dict:
    def to_api(s):
        return {
            "id": s.id,
            "health": s.health,
            "length": s.length,
            "body": [{"x": x, "y": y} for x, y in s.body],
        }
    snake = next(s for s in state.snakes if s.id == snake_id)
    return {
        "turn": state.turn,
        "you": to_api(snake),
        "board": {
            "width": state.width,
            "height": state.height,
            "food": [{"x": x, "y": y} for x, y in state.food],
            "hazards": [{"x": x, "y": y} for x, y in state.hazards],
            "snakes": [to_api(s) for s in state.snakes if s.alive],
        },
    }


def _random_agent(state: GameState, snake_id: str) -> str:
    moves = state.legal_moves(snake_id)
    return random.choice(moves) if moves else "down"


def _heuristic_agent(state: GameState, snake_id: str) -> str:
    from heuristic_agent import evaluate_move
    api = _state_to_api(state, snake_id)
    scores = {d: evaluate_move(d, api) for d in DIRECTIONS}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > float("-inf") else "down"


def _mcts_agent(state: GameState, snake_id: str, think: float = 0.2) -> str:
    # Re-root MCTS on this snake's perspective
    import mcts_agent as ma
    orig_id = state.my_id
    state = deepcopy(state)
    state.my_id = snake_id
    move = ma.mcts(state, think_time=think)
    return move


def _mcts_rave_agent(state: GameState, snake_id: str, think: float = 0.2) -> str:
    import mcts_rave_agent as ra
    state = deepcopy(state)
    state.my_id = snake_id
    return ra.mcts_rave(state, think_time=think)


AGENTS = {
    "random":    _random_agent,
    "heuristic": _heuristic_agent,
    "mcts":      _mcts_agent,
    "mcts_rave": _mcts_rave_agent,
}


# ── Game simulation ───────────────────────────────────────────────────────────

THINK_TIME = 0.15   # seconds per move per agent in tournament


def _make_initial_state(seed: int, game_idx: int) -> GameState:
    """Create a fresh 11×11 board with 4 snakes in corners."""
    rng = random.Random(seed * 1000 + game_idx)
    starts = [(9, 1), (9, 9), (1, 1), (1, 9)]
    rng.shuffle(starts)
    snakes = []
    for i, (x, y) in enumerate(starts):
        sid = f"snake_{i}"
        snakes.append(Snake(sid, [(x, y), (x, y), (x, y)], 100, 3))
    # Scatter food
    food = {(5, 5), (2, 0), (8, 10), (0, 5), (10, 5)}
    return GameState(
        width=11, height=11, turn=0,
        snakes=snakes, food=food, hazards=set(),
        my_id="snake_0",
    )


def run_game(agent_map: dict[str, str], seed: int, game_idx: int,
             max_turns: int = 300) -> dict[str, int]:
    """
    Simulate one game. agent_map: {snake_id: agent_name}.
    Returns {snake_id: survival_turn}.
    """
    state = _make_initial_state(seed, game_idx)
    # Track when each snake dies (later = better)
    death_turn: dict[str, int] = {s.id: 0 for s in state.snakes}

    for turn in range(max_turns):
        if len(state.alive_snakes()) <= 1:
            break

        moves: dict[str, str] = {}
        for snake in state.alive_snakes():
            agent_name = agent_map.get(snake.id, "random")
            agent_fn = AGENTS[agent_name]
            try:
                if agent_name in ("mcts", "mcts_rave"):
                    moves[snake.id] = agent_fn(state, snake.id, THINK_TIME)
                else:
                    moves[snake.id] = agent_fn(state, snake.id)
            except Exception:
                moves[snake.id] = "down"

        prev_alive = {s.id for s in state.alive_snakes()}
        state = state.step(moves)
        now_alive = {s.id for s in state.alive_snakes()}

        for sid in prev_alive - now_alive:
            death_turn[sid] = turn

    for snake in state.alive_snakes():
        death_turn[snake.id] = max_turns

    return death_turn


# ── Tournament ────────────────────────────────────────────────────────────────

def tournament(agent_names: list[str], n_games: int, seed: int):
    """
    Round-robin tournament: cycle agents across the 4 snake slots.
    Each game assigns one of each agent (if 4 agents) or repeats.
    """
    ratings = {name: trueskill.Rating() for name in set(agent_names)}
    results: dict[str, list[int]] = {n: [] for n in set(agent_names)}
    wins:    dict[str, int]       = {n: 0  for n in set(agent_names)}

    print(f"\nRunning {n_games} games with agents: {agent_names}")
    print("─" * 60)

    for game_idx in range(n_games):
        # Rotate agent assignment across snake slots each game
        assigned = [agent_names[i % len(agent_names)] for i in range(4)]
        random.shuffle(assigned)
        agent_map = {f"snake_{i}": assigned[i] for i in range(4)}
        slot_to_agent = {f"snake_{i}": assigned[i] for i in range(4)}

        t0 = time.time()
        death_turns = run_game(agent_map, seed, game_idx)
        elapsed = time.time() - t0

        # Determine rank (higher survival_turn = better rank)
        sorted_snakes = sorted(death_turns.items(), key=lambda x: -x[1])
        winner_id = sorted_snakes[0][0]
        winner_name = slot_to_agent[winner_id]
        wins[winner_name] += 1

        # TrueSkill update: pairwise comparisons (winner beats each other)
        # This handles the case where the same agent occupies multiple slots.
        winner_agent = slot_to_agent[sorted_snakes[0][0]]
        for _, (sid, _) in enumerate(sorted_snakes[1:], 1):
            loser_agent = slot_to_agent[sid]
            if winner_agent == loser_agent:
                continue  # can't rate same agent against itself
            new_w, new_l = trueskill.rate_1vs1(
                ratings[winner_agent], ratings[loser_agent]
            )
            ratings[winner_agent] = new_w
            ratings[loser_agent] = new_l

        for sid, dt in death_turns.items():
            results[slot_to_agent[sid]].append(dt)

        print(f"Game {game_idx+1:3d}/{n_games}  {elapsed:.1f}s  "
              f"winner={winner_name}  "
              f"turns={sorted([dt for dt in death_turns.values()])}")

    print("\n" + "═" * 60)
    print("FINAL LEADERBOARD")
    print("═" * 60)
    print(f"{'Agent':<14} {'μ':>6} {'σ':>6} {'Conservative':>14} {'Wins':>6} {'AvgTurn':>8}")
    print("─" * 60)

    leaderboard = []
    for name in set(agent_names):
        r = ratings[name]
        conservative = r.mu - 3 * r.sigma
        avg_turn = sum(results[name]) / len(results[name]) if results[name] else 0
        leaderboard.append((name, r, conservative, wins[name], avg_turn))

    leaderboard.sort(key=lambda x: -x[2])
    for rank, (name, r, cons, w, avg) in enumerate(leaderboard, 1):
        print(f"{rank}. {name:<12} {r.mu:>6.2f} {r.sigma:>6.2f} {cons:>14.2f} {w:>6} {avg:>8.1f}")
    print("─" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, __file__.replace("tournament.py", ""))
    parser = argparse.ArgumentParser(description="BattleSnake local tournament")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games to simulate (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--agents", nargs="+",
                        default=["random", "heuristic", "mcts", "mcts_rave"],
                        help="Agents to include (default: all four)")
    args = parser.parse_args()
    tournament(args.agents, args.games, args.seed)
