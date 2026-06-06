"""
grid.py: 2D grid world environment for the LLM agent.

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
TILE_FOG   = "?"

FOG_RADIUS = 4  # cells the agent can see around itself (Manhattan distance)

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
        """Return a deep copy of this state (safe for logging/replay without mutation)."""
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


def get_visible_cells(state: WorldState, radius: int) -> set:
    """
    Return the set of (col, row) tuples within Manhattan distance `radius`
    of the agent. Simple radius check - no raycast needed.
    Cells are included regardless of walls (wall cells themselves are visible).
    """
    ac, ar = state.agent_pos
    visible = set()
    for dc in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if abs(dc) + abs(dr) <= radius:
                nc, nr = ac + dc, ar + dr
                if 0 <= nc < state.width and 0 <= nr < state.height:
                    visible.add((nc, nr))
    return visible


def render_grid(state: WorldState, visible_cells: set = None) -> str:
    """Return an ASCII string of the current world.

    When visible_cells is provided, cells outside the set render as '?'
    (fog of war) - except the agent's own position is always visible.
    """
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

    # Apply fog of war if visible_cells provided
    if visible_cells is not None:
        for r in range(state.height):
            for c in range(state.width):
                if (c, r) not in visible_cells and (c, r) != (ac, ar):
                    grid[r][c] = TILE_FOG

    lines = ["  " + " ".join(str(c) for c in range(state.width))]
    for r, row in enumerate(grid):
        lines.append(f"{r} " + " ".join(row))
    return "\n".join(lines)


def get_observation(
    state: WorldState,
    fog_radius: int = None,
    discovered: dict = None,
) -> dict:
    """
    Build a structured observation dict describing everything
    the agent can perceive from its current position.

    When fog_radius is set:
      - known_items and known_goals are limited to visible cells.
      - fog_of_war / visible_radius fields are added.
      - discovered dict (mutable, caller-owned) is updated with newly seen
        items/goals and included as 'discovered_items' in the result.
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

    if fog_radius is not None:
        # Compute visible cells and filter items/goals
        visible = get_visible_cells(state, fog_radius)

        known_items = {
            item.name: {"col": c, "row": r, "symbol": item.symbol}
            for (c, r), item in state.items.items()
            if (c, r) in visible
        }

        known_goals = {
            goal_name: {"col": c, "row": r}
            for (c, r), goal_name in state.goals.items()
            if (c, r) in visible
        }

        # Update the discovered tracking dict
        if discovered is not None:
            if "items" not in discovered:
                discovered["items"] = {}
            if "goals" not in discovered:
                discovered["goals"] = {}
            discovered["items"].update(known_items)
            discovered["goals"].update(known_goals)

        discovered_items = dict(discovered["items"]) if discovered is not None else {}

        # Manhattan distance using only visible items
        nearest_distance = None
        if known_items:
            nearest_distance = min(
                abs(info["col"] - ac) + abs(info["row"] - ar)
                for info in known_items.values()
            )
        elif state.items:
            # Fall back to global distance so agent still has a hint
            nearest_distance = min(
                abs(ic - ac) + abs(ir - ar)
                for (ic, ir) in state.items
            )

        result = {
            "agent_position": {"col": ac, "row": ar},
            "inventory": list(state.agent_inventory),
            "surroundings": surroundings,
            "at_item": at_item.name if at_item else None,
            "at_goal": at_goal if at_goal else None,
            "known_items": known_items,
            "known_goals": known_goals,
            "nearest_item_distance": nearest_distance,
            "completed_goals": list(state.completed_goals),
            "steps_taken": state.steps,
            "fog_of_war": True,
            "visible_radius": fog_radius,
            "discovered_items": discovered_items,
        }
        return result

    # --- No fog: original behaviour ---
    known_items = {
        item.name: {"col": c, "row": r, "symbol": item.symbol}
        for (c, r), item in state.items.items()
    }

    known_goals = {
        goal_name: {"col": c, "row": r}
        for (c, r), goal_name in state.goals.items()
    }

    # Manhattan distance to the nearest remaining item (useful planning hint)
    nearest_distance = None
    if state.items:
        nearest_distance = min(
            abs(ic - ac) + abs(ir - ar)
            for (ic, ir) in state.items
        )

    return {
        "agent_position": {"col": ac, "row": ar},
        "inventory": list(state.agent_inventory),
        "surroundings": surroundings,
        "at_item": at_item.name if at_item else None,
        "at_goal": at_goal if at_goal else None,
        "known_items": known_items,
        "known_goals": known_goals,
        "nearest_item_distance": nearest_distance,
        "completed_goals": list(state.completed_goals),
        "steps_taken": state.steps,
    }