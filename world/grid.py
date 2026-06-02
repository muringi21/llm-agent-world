"""
grid.py:2D grid world environment for the LLM agent.

The world is a W x H tile grid. Each cell can contain:
  - Empty space
  - Wall
  - An item (key, gem, etc.)
  - A goal zone (drop-off point)
  - The agent itself

Coordinate system: (col, row) with (0,0) at top-left.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy


TILE_EMPTY = "."
TILE_WALL  = "#"
TILE_AGENT = "A"
TILE_GOAL  = "G"

DIRECTIONS = {
    "north": (0, -1),
    "south": (0,  1),
    "east":  (1,  0),
    "west":  (-1, 0),
}


@dataclass
class Item:
    name: str
    symbol: str


@dataclass
class WorldState:
    width: int
    height: int
    walls: set = field(default_factory=set)          # {(col, row), ...}
    items: dict = field(default_factory=dict)         # (col, row) -> Item
    goals: dict = field(default_factory=dict)         # (col, row) -> goal_name
    agent_pos: tuple = (1, 1)
    agent_inventory: list = field(default_factory=list)
    steps: int = 0
    completed_goals: list = field(default_factory=list)
    history: list = field(default_factory=list)       # list of (action, result) tuples

    def clone(self) -> "WorldState":
        s = copy.deepcopy(self)
        return s


def build_world(scenario: str = "delivery") -> WorldState:
    """
    Build a pre-designed world scenario.

    'delivery': Agent must pick up a KEY and bring it to the GOAL.
    'multi': Agent must collect two items and deliver them.
    """
    if scenario == "delivery":
        state = WorldState(width=8, height=8)
        # Outer walls
        for c in range(8):
            state.walls.add((c, 0))
            state.walls.add((c, 7))
        for r in range(8):
            state.walls.add((0, r))
            state.walls.add((7, r))
        # Interior walls — creates a simple maze feel
        for r in range(2, 5):
            state.walls.add((3, r))
        state.walls.add((3, 5))
        state.walls.add((5, 2))
        state.walls.add((5, 3))

        state.agent_pos = (1, 1)
        state.items[(5, 5)] = Item(name="key", symbol="K")
        state.goals[(6, 1)] = "deliver_key"
        return state

    elif scenario == "multi":
        state = WorldState(width=10, height=10)
        for c in range(10):
            state.walls.add((c, 0))
            state.walls.add((c, 9))
        for r in range(10):
            state.walls.add((0, r))
            state.walls.add((9, r))
        # Some interior walls
        for r in range(1, 7):
            state.walls.add((4, r))
        state.walls.discard((4, 3))  # gap in wall
        for c in range(5, 9):
            state.walls.add((c, 5))
        state.walls.discard((6, 5))  # gap

        state.agent_pos = (1, 1)
        state.items[(2, 7)] = Item(name="gem",    symbol="*")
        state.items[(7, 2)] = Item(name="crystal", symbol="C")
        state.goals[(8, 8)] = "deliver_gem_and_crystal"
        return state

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def render_grid(state: WorldState) -> str:
    """Return an ASCII string of the current world."""
    grid = [[TILE_EMPTY] * state.width for _ in range(state.height)]

    for (c, r) in state.walls:
        if 0 <= r < state.height and 0 <= c < state.width:
            grid[r][c] = TILE_WALL

    for (c, r), goal_name in state.goals.items():
        grid[r][c] = TILE_GOAL

    for (c, r), item in state.items.items():
        grid[r][c] = item.symbol

    ac, ar = state.agent_pos
    grid[ar][ac] = TILE_AGENT

    lines = ["  " + " ".join(str(c) for c in range(state.width))]
    for r, row in enumerate(grid):
        lines.append(f"{r} " + " ".join(row))
    return "\n".join(lines)


def get_observation(state: WorldState) -> dict:
    """
    Build a structured observation dict describing everything
    the agent can perceive from its current position.
    """
    ac, ar = state.agent_pos

    # What's in each adjacent cell
    surroundings = {}
    for direction, (dc, dr) in DIRECTIONS.items():
        nc, nr = ac + dc, ar + dr
        if (nc, nr) in state.walls:
            surroundings[direction] = "wall"
        elif (nc, nr) in state.items:
            surroundings[direction] = f"item:{state.items[(nc, nr)].name}"
        elif (nc, nr) in state.goals:
            surroundings[direction] = f"goal:{state.goals[(nc, nr)]}"
        elif 0 <= nc < state.width and 0 <= nr < state.height:
            surroundings[direction] = "empty"
        else:
            surroundings[direction] = "wall"

    # Items on current cell (just stepped onto)
    at_item = state.items.get((ac, ar))
    at_goal = state.goals.get((ac, ar))

    # All known item positions (agent has "memory" in this world)
    known_items = {
        item.name: {"col": c, "row": r, "symbol": item.symbol}
        for (c, r), item in state.items.items()
    }

    known_goals = {
        goal_name: {"col": c, "row": r}
        for (c, r), goal_name in state.goals.items()
    }

    return {
        "agent_position": {"col": ac, "row": ar},
        "inventory": list(state.agent_inventory),
        "surroundings": surroundings,
        "at_item": at_item.name if at_item else None,
        "at_goal": at_goal if at_goal else None,
        "known_items": known_items,
        "known_goals": known_goals,
        "completed_goals": list(state.completed_goals),
        "steps_taken": state.steps,
    }
