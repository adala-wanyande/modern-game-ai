import math
import time
import random
import typing
import argparse
from main_heuristic import heuristic_agent, start, end

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) representing a game state. Each node tracks its parent, children, the move that led to it, and statistics for wins and visits. The node also maintains a list of untried moves, 
    which are shuffled to ensure exploration diversity during expansion. The uct_select method implements the Upper Confidence Bound for Trees (UCT) formula to balance exploration and exploitation when selecting child nodes. 
    The update method updates the node's statistics based on the simulation results. 
    Inputs:
    - state (dict): The game state associated with this node.
    - parent (MCTSNode): The parent node in the search tree (default None).
    - move (str): The move that led to this node (default None).
    """
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = ["up", "down", "left", "right"]
        random.shuffle(self.untried_moves)

    def uct_select(self, exploration_constant=1.41):
        return max(self.children, key=lambda c: (c.wins / c.visits) + 
                   exploration_constant * math.sqrt(math.log(self.visits) / c.visits))

    def update(self, score):
        self.visits += 1
        self.wins += score

def rollout(state, max_rollout=80):
    """
    Simulates a random playout from the given state up tp the rollout depth or until a terminal state is reached. The function returns a heuristic score based on survival and length growth during the rollout.
    The survival component rewards the snake for staying alive for more turns, while the length component encourages growth and maintaining a longer body. The score is normalized to be between 0.0 and 1.0, 
    where higher scores indicate better outcomes for the snake. Contrary to the vanilla mcts rollout which only considers survival (+1) or loss (0), this enhanced rollout incorporates a more nuanced evaluation 
    of the snake's performance during the simulation, providing a richer signal for the MCTS algorithm to learn from.
    """
    if state is None: return 0
    
    curr = state
    steps = 0
    start_length = len(state['you']['body'])
    
    while steps < max_rollout:
        if is_terminal(curr):
            break
            
        valid_moves = []
        for m in ["up", "down", "left", "right"]:
            s = get_next_state(curr, m)
            if s is not None: 
                valid_moves.append(s)
            
        if not valid_moves: 
            break 
        
        #Random move selection
        curr = random.choice(valid_moves)
        steps += 1

    reward_score = reward(curr, steps, max_rollout, start_length)
    return reward_score
def rollout_heuristic(state, competitive: bool):
    """
    Simulates a playout guided by the heuristic agent from the given state up tp the rollout depth or until a terminal state is reached. The function returns a heuristic score based on survival and length growth during the rollout.
    The survival component rewards the snake for staying alive for more turns, while the length component encourages growth and maintaining a longer body. The score is normalized to be between 0.0 and 1.0, 
    where higher scores indicate better outcomes for the snake. Contrary to the vanilla mcts rollout which only considers survival (+1) or loss (0), this enhanced rollout incorporates a more nuanced evaluation 
    of the snake's performance during the simulation, providing a richer signal for the MCTS algorithm to learn from.
    """
    if state is None: return 0
    
    curr = state
    steps = 0
    max_rollout = 80 
    start_length = len(state['you']['body'])
    
    while steps < max_rollout:
        if is_terminal(curr):
            break
            
        # Instead of random moves, use the heuristic agent to guide the rollout
        if competitive:
            heuristic_result = heuristic_agent(curr, competitive=competitive, mcts_use=True)
        else:
            heuristic_result = heuristic_agent(curr, competitive=False, mcts_use=True)
        move = heuristic_result["move"]
        
        next_s = get_next_state(curr, move)
        if next_s is None: 
            break 
            
        curr = next_s
        steps += 1
    reward_score = reward(curr, steps, max_rollout, start_length)
    
    return reward_score

def reward(curr, steps, max_rollout, start_length):
    """
    Calculates a heuristic reward score based on the final state of the rollout, the number of steps taken, and the initial length of the snake. The reward is designed to encourage both survival and growth during the rollout.
    The survival component provides a significant portion of the reward, incentivizing the snake to stay alive for as long as possible. The length component rewards the snake for maintaining a longer body and for growth
    during the rollout, which can be crucial for success in Battlesnake. The final score is normalized to be between 0.0 and 1.0, providing a consistent scale for the MCTS algorithm to evaluate different moves based on their simulated outcomes.
    Inputs:
    - curr (dict): The final game state at the end of the rollout.
    - steps (int): The number of steps taken during the rollout.
    - max_rollout (int): The maximum number of steps allowed in the rollout.
    - start_length (int): The initial length of the snake at the start of the rollout.
    Returns:
    - float: A heuristic reward score between 0.0 and 1.0, where higher scores indicate better outcomes for the snake.
    """
    if not is_terminal(curr):
        survival_reward = 0.8
    else:
        survival_reward = (steps / max_rollout) * 0.4
    
    current_length = len(curr['you']['body'])
    growth = current_length - start_length
    length_reward = min(current_length / 20, 1.0) * 0.15 
    growth_reward = min(growth, 5) / 5 * 0.05
    return max(0, min(1.0, survival_reward + length_reward + growth_reward))

def get_next_state(game_state, my_move):
    """
    Get the next game state resulting from applying the given move to the current game state. The function checks for collisions with walls, hazards, and other snakes, as well as food consumption and health updates.
    If the move results in a collision or the snake's health drops to zero or below, the function returns None, indicating a terminal state. Otherwise, it returns the new game state after applying the move,
    including the updated position of the snake's head, body, and health. This function is crucial for simulating potential future states during the MCTS algorithm's expansion and simulation phases. 

    This is different than the vanilla mcts, here we consider stationary oponents and hazards, which allows us to prune "Death" states early in the search tree, improving efficiency and decision quality.
    Inputs:
    - game_state (dict): The current game state provided by the Battlesnake engine.
    - my_move (str): The move to apply ("up", "down", "left", "right").
    Returns:
    - dict or None: The new game state after applying the move, or None if the move results in a terminal state (collision or death).
    """
    if game_state is None: return None
    
    # Fast shallow copy
    new_state = game_state.copy()
    my_snake = game_state['you'].copy()
    my_snake['body'] = list(game_state['you']['body'])
    new_state['you'] = my_snake
    
    head = my_snake['head']
    if my_move == "up":    new_head_tup = (head["x"], head["y"] + 1)
    elif my_move == "down":  new_head_tup = (head["x"], head["y"] - 1)
    elif my_move == "left":  new_head_tup = (head["x"] - 1, head["y"])
    else:                    new_head_tup = (head["x"] + 1, head["y"])

    new_head_dict = {"x": new_head_tup[0], "y": new_head_tup[1]}

    # Wall Collision
    if (new_head_tup[0] < 0 or new_head_tup[0] >= game_state['board']['width'] or 
        new_head_tup[1] < 0 or new_head_tup[1] >= game_state['board']['height']):
        return None

    # Obstacle Collision
    if new_head_tup in game_state['_obstacle_set']:
        if any(seg == new_head_dict for seg in my_snake['body'][:-1]):
            return None
        for snake in game_state['board']['snakes']:
            if snake['id'] == my_snake['id']: continue
            if new_head_dict == snake['head'] and len(my_snake['body']) <= len(snake['body']):
                return None
            elif any(seg == new_head_dict for seg in snake['body']):
                return None

    # Hazard/Health
    if new_head_tup in game_state['_hazard_set']:
        my_snake['health'] -= 15 
    else:
        my_snake['health'] -= 1
    
    if my_snake['health'] <= 0: return None

    # Food
    if new_head_tup in game_state['_food_set']:
        my_snake['health'] = 100
    else:
        my_snake['body'].pop()

    my_snake['body'].insert(0, new_head_dict)
    my_snake['head'] = new_head_dict
    new_state['turn'] = game_state.get('turn', 0) + 1
    return new_state

def is_terminal(state):
    """
    Checks if the given game state is terminal, meaning the snake has collided with a wall, hazard, or another snake, or if its health has dropped to zero. 
    The function returns True if the state is terminal (death) and False otherwise.
    """
    if state is None: return True
    my_snake = state['you']
    head = my_snake['head']
    if (head['x'] < 0 or head['x'] >= state['board']['width'] or 
        head['y'] < 0 or head['y'] >= state['board']['height']):
        return True
    if my_snake['health'] <= 0 or state.get('turn', 0) >= 300:
        return True
    return False


def mcts_agent(game_state, heuristic=False, competitive=False, exploration_constant=1.41, max_rollout=80) -> typing.Dict:
    """
    An implementation of a Monte Carlo Tree Search (MCTS) agent for Battlesnake. The agent builds a search tree of game states, simulates random playouts to evaluate potential moves, 
    and selects the move that leads to the most promising outcomes based on visit counts. The MCTS agent is operating within an 800ms time budget. The final move decision is based on 
    the child node with the highest visit count, which indicates the most explored and potentially best move.
    The agent consists of the following main components:
    1. Selection: Traverses the tree using the UCT formula to select child nodes until it reaches a node with untried moves or a terminal state.
    2. Expansion: If the selected node has untried moves, it expands the tree by creating a new child node for one of the untried moves.
    3. Simulation: Performs playout from the new node until a rollout depth is reached to evaluate the outcome of that move. When the heuristic flag is set to True, the simulation is guided by a heuristic agent instead of random moves,
                   which can lead to more informed evaluations and better performance.
    4. Backpropagation: Updates the statistics of the nodes along the path from the new node back to the root based on the simulation results.
    Inputs:
    - game_state (dict): The current game state provided by the Battlesnake engine.
    Returns:
    - dict: The chosen move direction based on the MCTS algorithm.
    """
    game_state['_food_set'] = set((f['x'], f['y']) for f in game_state['board']['food'])
    game_state['_hazard_set'] = set((h['x'], h['y']) for h in game_state['board']['hazards'])
    game_state['_obstacle_set'] = set((p['x'], p['y']) for s in game_state['board']['snakes'] for p in s['body'])

    root = MCTSNode(game_state)
    start_time = time.time()
    
    # 800ms Time Budget
    iterations = 0
    while time.time() - start_time < 0.1:
        node = root
        # 1. Selection
        while node.children and not node.untried_moves:
            node = node.uct_select(exploration_constant=exploration_constant)
            if node.state is None: break

        # 2. Expansion
        if node.untried_moves and node.state is not None:
            move = node.untried_moves.pop()
            next_state = get_next_state(node.state, move)
            new_node = MCTSNode(next_state, parent=node, move=move)
            node.children.append(new_node)
            node = new_node

        # 3. Simulation
        score = rollout(node.state, max_rollout=max_rollout)

        # 4. Backpropagation
        curr = node
        while curr:
            curr.update(score)
            curr = curr.parent

    #print(f"MCTS completed {iterations} iterations.")
    
    if not root.children:
        return {"move": "down"}

    # Final Decision: Choose child with highest visit count
    valid_children = [c for c in root.children if c.state is not None]
    if not valid_children:
        for m in ["up", "down", "left", "right"]:
            if get_next_state(game_state, m) is not None: return {"move": m}
        return {"move": "up"}
    best_child = max(valid_children, key=lambda c: c.visits)
    return {"move": best_child.move}

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Team 13",
        "color": getattr(info, "snake_color", "#695DD1"),
        "name": getattr(info, "snake_name", "MCTS-Snake"),
        "head": "shades",
        "tail": "bolt",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", type=str, default="MCTS-Snake")
    parser.add_argument("--color", type=str, default="#695DD1")
    args = parser.parse_args()
    info.snake_color = args.color
    info.snake_name = args.name
    from server import run_server
    # Use a lambda to wrap the call so the server can inject 'game_state' later
    run_server({
        "info": info, 
        "start": start, 
        "move": lambda gs: mcts_agent(gs, heuristic=False, competitive=False), 
        "end": end
    }, port=args.port)