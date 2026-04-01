# MGAIA Assignment 2: Battlesnakes

This assignment were made in the context of the Modern Game AI in Algorithms course. Here we provide the source code for a Battlesnake template written in python for which aim is to win the game and have the snake that survives the longest. Using as a starting point the starter-snake-python.zip file provided in the assignment we implement a Heuristic and a Monte Carlo Tree Search agent. In the former case we expore two different cases for competitive and friendly mode, in which on the first case we prioritize food that is also located further from the opponents. In the MCTS agent we implemented a random rollout (vanilla MCTS) as well as a rollout guided by our heuristic agent. 

## Technologies Used

This project uses [Python 3](https://www.python.org/), [Flask](https://flask.palletsprojects.com/) and the latest version of Go.

Install dependencies using pip

```sh
pip install -r requirements.txt
```

## Run Your Battlesnake
To run a single battlesnake you need to comment out the PORTS in run_game.py. The current structure assumes that 4 battlesnakes exist. 

### Heuristic agent
We implemented a heuristic agent to select the best moves for our snakes. In essence a heuristic agent selcts the move based on a set of rules on a statonary state of the game. In this assignment we attempted to implement two approaches for the heuristic agent: one friendly (the snakes search for food only at low HP), and one competitive (the snakes prioritize fod early on and focus on food locations furthest from the opponents).
To run the first case you can open 4 different terminals from this repository and run:
```sh
 python main_heuristic.py --port 8000 --name "Friendly 1"
```
```sh
 python main_heuristic.py --port 8001 --name "Friendly 2"
```
```sh
 python main_heuristic.py --port 8002 --name "Friendly 3"
```
```sh
 python main_heuristic.py --port 8003 --name "Friendly 4"
```
In the competitive mode you can run the following:

```sh
  python main_heuristic.py --port 8000 --name "Competitive 1" --competitive
```
```sh
 python main_heuristic.py --port 8001 --name "Competitive 2" --competitive
```
```sh
 python main_heuristic.py --port 8002 --name "Competitive 3" --competitive
```
```sh
 python main_heuristic.py --port 8003 --name "Competitive 4" --competitive
```
Ofcourse, at this stage you can run every possible combination of heuristic friendly and competitive snakes. 
 
To run the game in a seperate terminal from this repositiory you can run:
```sh
  python run_game.py
```
Make sure that all ports in the run game.py file are at that time running a snake program, otherwise it will throw an error.

### MCTS agent
We implement an MCTS agent to improve our performance. Fist we implement the vanilla MCTS with a random rollout and then we use the heuristic agent to guide our simulations steps. 
To run the first case you can open 4 different terminals from this repository and run:
```sh
 python main_mcts.py --port 8000 --name "MCTS 1"
```
```sh
 python main_mcts.py --port 8001 --name "MCTS 1"
```
```sh
 python main_mcts.py --port 8002 --name "MCTS 1"
```
```sh
 python main_mcts.py --port 8003 --name "MCTS 1"
```
For testing the heuristic rollout you can run:

```sh
  python main_MCTS_heuristic.py --port 8000 --name "MCTS 1 Friendly" 
```
```sh
  python main_MCTS_heuristic.py --port 8001 --name "MCTS 2 Friendly" 
```
```sh
  python main_MCTS_heuristic.py --port 8002 --name "MCTS 3 Friendly" 
```
```sh
  python main_MCTS_heuristic.py --port 8003 --name "MCTS 4 Friendly" 
```
Or for the competitive mode use similarly:
```sh
  python main_MCTS_heuristic.py --port 8000 --name "MCTS 1 Competitive" --competitive
```

Of course, again, you can run every possible combination of heuristic (Friendly, Competitive), vanilla MCTS, or heuristic MCTS (Friendly, Competitive) agent to evaluate the performance of the snakes. 
In order to run the game, open a seperate terminal from this repository and run:
```sh
  python run_game.py
```

### Hyper parameter exploration

We explore the effect of the exploration constant Cp and rollout depth by testing the vanilla MCTS agent against 3 heuristic ones and varying the parameter of interest. By running the following command we simulate a number of battlesnake games, for which each game includes one MCTS agent and 3 Heuristic ones. The code keeps track of the average turns, wining rates as well as the trueskill score, saves the results in a npz file, and plots them as well in the same folder. 

For testing the Cp by running 1000 games you can run in terminal:
```sh
  python tournament.py --games 1000 --test Cp   
```

For testing the rollout depth by running 1000 games you can run in terminal:
```sh
  python tournament.py --games 1000 --test depth   
```