import argparse
import random
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import trueskill

from game_simulator import GameState, Snake, DIRECTIONS
from main_heuristic import heuristic_agent
import main_MCTS as ma 

# ── Agent interfaces ──────────────────────────────────────────────────────────
def _state_to_api(state: GameState, snake_id: str) -> dict:
    """Converts GameState object to the dict format mcts_agent.py expects."""
    def to_api(s):
        return {
            "id": s.id,
            "health": s.health,
            "length": s.length,
            "body": [{"x": x, "y": y} for x, y in s.body],
            "head": {"x": s.body[0][0], "y": s.body[0][1]}
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

def _mcts_agent_configured(state: GameState, snake_id: str, config: dict) -> str:
    api_dict = _state_to_api(state, snake_id)
    result = ma.mcts_agent(
        api_dict, 
        heuristic=False, 
        competitive=False,
        exploration_constant=config.get('Cp', 1.41),
        max_rollout=config.get('depth', 80)
    )
    return result["move"]

def _heuristic_agent_wrapper(state: GameState, snake_id: str):
    """Wrapper to make the heuristic_agent compatible with the simulator."""
    api_dict = _state_to_api(state, snake_id)
    result = heuristic_agent(api_dict, competitive=True, mcts_use=True)
    return result["move"]

# ── Simulation Core ───────────────────────────────────────────────────────────

def _make_initial_state(seed: int, game_idx: int) -> GameState:
    rng = random.Random(seed * 1000 + game_idx)
    starts = [(1, 1), (1, 9), (9, 1), (9, 9)]
    rng.shuffle(starts)
    snakes = []
    for i, (x, y) in enumerate(starts):
        sid = f"snake_{i}"
        snakes.append(Snake(sid, [(x, y)]*3, 100, 3))
    
    food = {(5, 5), (2, 2), (8, 8), (2, 8), (8, 2)}
    return GameState(
        width=11, height=11, turn=0,
        snakes=snakes, food=food, hazards=set(),
        my_id="snake_0",
    )

def run_game_vs_heuristic(mcts_config, seed, game_idx):
    state = _make_initial_state(seed, game_idx)
    death_turns = {s.id: 0 for s in state.snakes}
    
    for turn in range(400):
        alive = state.alive_snakes()
        if len(alive) <= 1: break

        moves = {}
        for snake in alive:
            if snake.id == "snake_0":
                moves[snake.id] = _mcts_agent_configured(state, snake.id, mcts_config)
            else:
                moves[snake.id] = _heuristic_agent_wrapper(state, snake.id)

        prev_alive = {s.id for s in alive}
        state = state.step(moves)
        for sid in prev_alive - {s.id for s in state.alive_snakes()}:
            death_turns[sid] = turn

    for s in state.alive_snakes():
        death_turns[s.id] = 400
    return death_turns

def run_hyperparameter_exploration_tournament(n_games, seed, test_type="Cp"):
    output_dir = f"tournament_{test_type}_results"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    num_unique_values = 10 
    if test_type == "Cp":
        # Mode 1: Vary Cp, keep depth baseline
        test_points = np.linspace(0.1, 10.0, num=num_unique_values).tolist()
        param_name = "Cp"
        base_config = {"Cp": 1.41, "depth": 80}
        print(f"\nMODE: Exploring Exploration Constant (Cp)")
    else:
        # Mode 2: Vary Depth, lock Cp at the optimized 5.6
        test_points = [20, 50, 80, 120, 180, 250, 350, 500, 750, 1000]
        num_unique_values = len(test_points)
        param_name = "depth"
        base_config = {"Cp": 8, "depth": 80}
        print(f"\n MODE: Optimizing Rollout Depth (Fixed Cp: 8)")

    # Tracking MCTS specifically
    survival_stats = {v: [] for v in test_points}
    win_counts = {v: 0 for v in test_points}
    ts_ratings = {v: trueskill.Rating() for v in test_points}
    heuristic_rating = trueskill.Rating() 
    
    env = trueskill.TrueSkill(draw_probability=0)

    print(f"Total Games: {n_games} (3x Heuristic vs 1x MCTS)")
    print("─" * 75)

    for g in range(n_games):
        current_val = test_points[g % num_unique_values]
        conf = base_config.copy()
        conf[param_name] = current_val

        results = run_game_vs_heuristic(conf, seed, g)
        
        # Rank by survival descending
        ranked_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # TrueSkill ranking
        team_ratings = []
        for sid, _ in ranked_items:
            if sid == "snake_0":
                team_ratings.append((ts_ratings[current_val],))
            else:
                team_ratings.append((heuristic_rating,))
        
        new_ratings = env.rate(team_ratings, ranks=[0, 1, 2, 3])
        
        # Update ratings & stats
        max_t = max(results.values())
        for i, (sid, turns) in enumerate(ranked_items):
            if sid == "snake_0":
                ts_ratings[current_val] = new_ratings[i][0]
                survival_stats[current_val].append(turns)
                if turns == max_t:
                    win_counts[current_val] += 1
            else:
                heuristic_rating = new_ratings[i][0]

        print(f"Game {g+1}/{n_games} ({param_name}: {current_val}) - MCTS Rank: {[s[0] for s in ranked_items].index('snake_0')+1}")

    # 3. Process Final Data
    sorted_test_points = sorted(test_points)
    final_vals, final_avgs, final_stds, final_win_rates = [], [], [], []
    final_ts_mu, final_ts_sigma, final_ts_conservative = [], [], []

    print("\n" + "═" * 95)
    print(f"STATISTICAL SUMMARY: {test_type.upper()} TEST")
    print("═" * 95)
    print(f"{'Value':<8} | {'N':<4} | {'Avg Surv':<10} | {'Win%':<8} | {'Mu':<7} | {'Sigma':<7} | {'TrueSkill'}")
    print("-" * 95)

    for v in sorted_test_points:
        data = np.array(survival_stats[v])
        n_samples = len(data)
        avg = np.mean(data) if n_samples > 0 else 0
        std = np.std(data) if n_samples > 0 else 0
        win_rate = (win_counts[v] / n_samples * 100) if n_samples > 0 else 0
        
        r = ts_ratings[v]
        conservative = r.mu - (3 * r.sigma)
        
        final_vals.append(v)
        final_avgs.append(avg)
        final_stds.append(std)
        final_win_rates.append(win_rate)
        final_ts_mu.append(r.mu)
        final_ts_sigma.append(r.sigma)
        final_ts_conservative.append(conservative)
        
        print(f"{v:>8.2f} | {n_samples:>4} | {avg:>10.1f} | {win_rate:>7.1f}% | {r.mu:>7.2f} | {r.sigma:>7.2f} | {conservative:>10.2f}")

    # 4. Save and Plot
    np.savez(os.path.join(output_dir, f"{test_type}_results_1.npz"), 
             vals=final_vals, avgs=final_avgs, wins=final_win_rates, ts=final_ts_conservative)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    ax1.errorbar(final_vals, final_avgs, yerr=final_stds, fmt='-o', ecolor='gray', color='blue', alpha=0.7)
    ax1.set_ylabel("Avg Survival (Turns)")
    ax1.set_title(f"MCTS vs Heuristic: {test_type} Analysis")

    ax2.plot(final_vals, final_win_rates, '-o', color='green')
    ax2.set_ylabel("Win Rate (%)")

    ax3.plot(final_vals, final_ts_conservative, '-o', color='purple', label="MCTS Variant")
    # Add the heuristic baseline for comparison
    h_baseline = heuristic_rating.mu - (3 * heuristic_rating.sigma)
    ax3.axhline(y=h_baseline, color='r', linestyle='--', label=f'Heuristic Baseline ({h_baseline:.1f})')
    ax3.set_ylabel("TrueSkill Rating")
    ax3.set_xlabel(f"{test_type} Value")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_type}_metrics_plot_1.png"))
    plt.close()
    print(f"\n Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100, help="Total games to run")
    parser.add_argument("--test", choices=["Cp", "depth"], default="Cp", help="Parameter to vary")
    args = parser.parse_args()
    
    run_hyperparameter_exploration_tournament(args.games, 42, test_type=args.test)