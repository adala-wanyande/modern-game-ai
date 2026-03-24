"""
Battlesnake board simulator for MCTS rollouts.

Rules implemented (standard + hz_hazard_pits):
  - Health decreases 1/turn; +hazard_damage on hazard head entry
  - Food restores health to 100 and grows snake by 1
  - Snake eliminated when: health <= 0, out-of-bounds, body/wall collision,
    head-to-head vs equal/larger snake
  - Hazard pits appear from turn 26, stacking every 25 turns (4 stacks),
    each stack adds 14 damage; fully stacked after turn 101; drains at turn 176
  - Min 2 food on board; 25% spawn chance each turn
"""

import copy
import random
from collections import deque
from typing import Optional


DIRECTIONS = ("up", "down", "left", "right")
DELTA = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def _add(pos: tuple, d: str) -> tuple:
    dx, dy = DELTA[d]
    return (pos[0] + dx, pos[1] + dy)


def _manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Snake:
    def __init__(self, id_: str, body: list[tuple], health: int, length: int):
        self.id = id_
        self.body = list(body)   # list of (x,y) tuples, body[0] = head
        self.health = health
        self.length = length
        self.alive = True
        self.ate_food = False

    @property
    def head(self) -> tuple:
        return self.body[0]

    def copy(self) -> "Snake":
        s = Snake(self.id, list(self.body), self.health, self.length)
        s.alive = self.alive
        s.ate_food = self.ate_food
        return s


class GameState:
    """
    Lightweight game state for MCTS simulation.
    Can be initialised from a raw Battlesnake API dict or copied.
    """

    def __init__(self, width: int, height: int, turn: int,
                 snakes: list[Snake], food: set[tuple],
                 hazards: set[tuple], my_id: str):
        self.width = width
        self.height = height
        self.turn = turn
        self.snakes: list[Snake] = snakes
        self.food: set[tuple] = food
        self.hazards: set[tuple] = hazards
        self.my_id = my_id

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_api(cls, game_state: dict) -> "GameState":
        board = game_state["board"]
        my_id = game_state["you"]["id"]
        snakes = []
        for s in board["snakes"]:
            body = [(seg["x"], seg["y"]) for seg in s["body"]]
            snakes.append(Snake(s["id"], body, s["health"], s["length"]))
        food = {(f["x"], f["y"]) for f in board["food"]}
        hazards = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        return cls(
            width=board["width"],
            height=board["height"],
            turn=game_state["turn"],
            snakes=snakes,
            food=food,
            hazards=hazards,
            my_id=my_id,
        )

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def my_snake(self) -> Optional[Snake]:
        for s in self.snakes:
            if s.id == self.my_id:
                return s
        return None

    def is_terminal(self) -> bool:
        me = self.my_snake
        return me is None or not me.alive

    def alive_snakes(self) -> list[Snake]:
        return [s for s in self.snakes if s.alive]

    def _body_set(self, exclude_tails: bool = True) -> set[tuple]:
        occupied = set()
        for s in self.snakes:
            if not s.alive:
                continue
            end = -1 if exclude_tails else len(s.body)
            for seg in s.body[:end]:
                occupied.add(seg)
        return occupied

    def legal_moves(self, snake_id: str) -> list[str]:
        snake = next((s for s in self.snakes if s.id == snake_id), None)
        if snake is None or not snake.alive:
            return []
        occupied = self._body_set()
        moves = []
        for d in DIRECTIONS:
            nx, ny = _add(snake.head, d)
            if (0 <= nx < self.width and 0 <= ny < self.height
                    and (nx, ny) not in occupied):
                moves.append(d)
        return moves or ["down"]  # must return something

    def flood_fill(self, start: tuple) -> int:
        occupied = self._body_set()
        if start in occupied:
            return 0
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and (nx, ny) not in visited
                        and (nx, ny) not in occupied):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return len(visited)

    # ── Hazard computation ───────────────────────────────────────────────────

    def _hazard_damage(self) -> int:
        """Damage per turn from hazard pits (head entry only)."""
        if self.turn < 26:
            return 0
        stacks = min(4, (self.turn - 26) // 25 + 1)
        if self.turn >= 176:
            # Draining: 75 turns fully stacked → drain
            return 0
        base = 14
        return base + (stacks - 1)

    # ── Simulation step ──────────────────────────────────────────────────────

    def step(self, moves: dict[str, str]) -> "GameState":
        """
        Advance one turn given a moves dict {snake_id: direction}.
        Returns a new GameState (does not mutate self).
        """
        new_snakes = [s.copy() for s in self.snakes]
        new_food = set(self.food)

        # 1. Move all snakes
        for s in new_snakes:
            if not s.alive:
                continue
            direction = moves.get(s.id, random.choice(DIRECTIONS))
            new_head = _add(s.head, direction)
            s.body.insert(0, new_head)
            s.health -= 1
            s.ate_food = False

        # 2. Check food consumption (before tail removal)
        for s in new_snakes:
            if not s.alive:
                continue
            if s.head in new_food:
                new_food.remove(s.head)
                s.health = 100
                s.length += 1
                s.ate_food = True

        # 3. Remove tails (don't remove if snake ate food)
        for s in new_snakes:
            if not s.alive:
                continue
            if not s.ate_food:
                s.body.pop()

        # 4. Compute occupied cells for collision detection
        head_positions: dict[tuple, list[Snake]] = {}
        for s in new_snakes:
            if s.alive:
                head_positions.setdefault(s.head, []).append(s)

        body_cells: set[tuple] = set()
        for s in new_snakes:
            if s.alive:
                for seg in s.body[1:]:
                    body_cells.add(seg)

        # 5. Eliminate snakes
        hazard_dmg = self._hazard_damage()
        new_hazards = self._next_hazards()

        for s in new_snakes:
            if not s.alive:
                continue
            x, y = s.head
            # Out of bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                s.alive = False
                continue
            # Health depleted
            if s.health <= 0:
                s.alive = False
                continue
            # Body collision
            if s.head in body_cells:
                s.alive = False
                continue
            # Hazard damage (head entry)
            if s.head in new_hazards and hazard_dmg > 0:
                s.health -= hazard_dmg
                if s.health <= 0:
                    s.alive = False
                    continue

        # 6. Head-to-head collisions
        for pos, colliders in head_positions.items():
            if len(colliders) < 2:
                continue
            max_len = max(s.length for s in colliders)
            for s in colliders:
                if s.length < max_len:
                    s.alive = False
                elif s.length == max_len:
                    s.alive = False  # equal length → both die

        # 7. Food spawning (25% chance, min 2)
        remaining_food = len(new_food)
        if remaining_food < 2:
            new_food |= self._spawn_food(2 - remaining_food, new_snakes, new_food)
        elif random.random() < 0.25:
            new_food |= self._spawn_food(1, new_snakes, new_food)

        return GameState(
            width=self.width,
            height=self.height,
            turn=self.turn + 1,
            snakes=new_snakes,
            food=new_food,
            hazards=new_hazards,
            my_id=self.my_id,
        )

    def _next_hazards(self) -> set[tuple]:
        """Recompute hazard set for next turn (simplified: ring border expansion)."""
        # For simulation purposes we keep hazards static (they expand slowly)
        return set(self.hazards)

    def _spawn_food(self, n: int, snakes: list[Snake],
                    existing_food: set[tuple]) -> set[tuple]:
        occupied = set(existing_food)
        for s in snakes:
            if s.alive:
                occupied.update(s.body)
        empty = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not empty:
            return set()
        sample = random.sample(empty, min(n, len(empty)))
        return set(sample)

    # ── Copy ─────────────────────────────────────────────────────────────────

    def copy(self) -> "GameState":
        return GameState(
            width=self.width,
            height=self.height,
            turn=self.turn,
            snakes=[s.copy() for s in self.snakes],
            food=set(self.food),
            hazards=set(self.hazards),
            my_id=self.my_id,
        )
