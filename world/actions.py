"""
actions.py: Defines the action space and applies actions to world state.

Available actions:
  move_north / move_south / move_east / move_west
  pick_up
  drop <item_name>
  wait

Each action returns (new_state, result_message).
"""

from __future__ import annotations
from world.grid import WorldState, Item, DIRECTIONS
import copy


ActionResult = tuple[WorldState, str]


def apply_action(state: WorldState, action: str) -> ActionResult:
    """
    Parse and apply an action string to the world state.
    Returns (new_state, result_message).
    Does NOT mutate the original state.
    """
    s = copy.deepcopy(state)
    s.steps += 1
    action = action.strip().lower()

    # Movement
    for direction, (dc, dr) in DIRECTIONS.items():
        if action == f"move_{direction}":
            return _move(s, dc, dr, direction)

    if action == "pick_up":
        return _pick_up(s)

    if action.startswith("drop"):
        parts = action.split(maxsplit=1)
        item_name = parts[1] if len(parts) > 1 else ""
        return _drop(s, item_name)

    if action == "wait":
        s.history.append(("wait", "Agent waited."))
        return s, "Agent waited."

    s.history.append((action, f"Unknown action: '{action}'"))
    return s, f"Unknown action: '{action}'. Valid actions: move_north, move_south, move_east, move_west, pick_up, drop <item>, wait."


def _move(state: WorldState, dc: int, dr: int, direction: str) -> ActionResult:
    ac, ar = state.agent_pos
    nc, nr = ac + dc, ar + dr

    if (nc, nr) in state.walls:
        msg = f"Blocked — wall to the {direction}."
        state.history.append((f"move_{direction}", msg))
        return state, msg

    if not (0 <= nc < state.width and 0 <= nr < state.height):
        msg = f"Blocked — boundary to the {direction}."
        state.history.append((f"move_{direction}", msg))
        return state, msg

    state.agent_pos = (nc, nr)
    msg = f"Moved {direction} to ({nc}, {nr})."

    # Describe what's now at this cell
    if (nc, nr) in state.items:
        item = state.items[(nc, nr)]
        msg += f" You see a {item.name} here."
    if (nc, nr) in state.goals:
        goal_name = state.goals[(nc, nr)]
        msg += f" You are at the goal zone."
        if goal_name.startswith("reach_") and goal_name not in state.completed_goals:
            state.completed_goals.append(goal_name)
            msg += f" Goal '{goal_name}' completed!"

    state.history.append((f"move_{direction}", msg))
    return state, msg


def _pick_up(state: WorldState) -> ActionResult:
    pos = state.agent_pos
    if pos not in state.items:
        msg = "Nothing to pick up here."
        state.history.append(("pick_up", msg))
        return state, msg

    item = state.items.pop(pos)
    state.agent_inventory.append(item.name)
    msg = f"Picked up {item.name}. Inventory: {state.agent_inventory}."
    state.history.append(("pick_up", msg))
    return state, msg


def _drop(state: WorldState, item_name: str) -> ActionResult:
    if item_name not in state.agent_inventory:
        msg = f"You don't have '{item_name}' in your inventory. Inventory: {state.agent_inventory}."
        state.history.append((f"drop {item_name}", msg))
        return state, msg

    pos = state.agent_pos

    # Check if this completes a goal
    goal = state.goals.get(pos)
    if goal:
        state.agent_inventory.remove(item_name)
        state.completed_goals.append(goal)
        msg = f"Dropped {item_name} at goal zone. Goal '{goal}' completed!"
        state.history.append((f"drop {item_name}", msg))
        return state, msg

    # Just drop on the floor
    state.agent_inventory.remove(item_name)
    state.items[pos] = Item(name=item_name, symbol=item_name[0].upper())
    msg = f"Dropped {item_name} at ({pos[0]}, {pos[1]})."
    state.history.append((f"drop {item_name}", msg))
    return state, msg


def get_action_space_description() -> str:
    return """
Available actions (respond with EXACTLY one per turn):
  move_north   — move one step north (row - 1)
  move_south   — move one step south (row + 1)
  move_east    — move one step east  (col + 1)
  move_west    — move one step west  (col - 1)
  pick_up      — pick up the item at your current position
  drop <name>  — drop a named item (e.g. "drop key") — works at goal zone to score
  wait         — do nothing this step
""".strip()
