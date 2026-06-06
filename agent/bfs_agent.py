"""
bfs_agent.py: BFS(breadth-first search) deterministic agent.

Solves delivery/multi scenarios optimally (fewest moves) without using an LLM.
Used as a baseline to compare against the LLM agent.

Interface is identical to LLMAgent: choose_action(state: WorldState) -> str
"""

from __future__ import annotations
from collections import deque
from world.grid import WorldState, DIRECTIONS


class BFSAgent:
    """
    Deterministic BFS pathfinding agent.

    Strategy:
      1. If carrying an item, navigate to the goal zone and drop it.
      2. Otherwise, find the nearest item (by BFS path length), navigate to it,
         and pick it up.
      3. Repeat until all goals are completed.
    """

    def __init__(self):
        self._path: list[str] = []  # queued action sequence

    def choose_action(self, state: WorldState) -> str:
        """
        Return the next action to take given the current world state.
        Recalculates BFS plan whenever the queue is empty or stale.
        """
        # Execute next queued action if available
        if self._path:
            return self._path.pop(0)

        # Replan
        self._path = self._plan(state)
        if self._path:
            return self._path.pop(0)

        return "wait"

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, state: WorldState) -> list[str]:
        """
        Compute the next sequence of actions to make progress.

        Priority:
          1. If standing on an item — pick it up immediately.
          2. If carrying items — go to goal and drop each one.
          3. Otherwise — BFS to the nearest item.
        """
        ac, ar = state.agent_pos

        # 1. Standing on an item?
        if (ac, ar) in state.items:
            return ["pick_up"]

        # 2. Carrying something — navigate to goal and drop
        if state.agent_inventory:
            item_to_drop = state.agent_inventory[0]
            # Find any goal position
            if state.goals:
                goal_pos = next(iter(state.goals.keys()))
                path = self._bfs(state, (ac, ar), goal_pos)
                if path is not None:
                    # Append the drop action at the end
                    return path + [f"drop {item_to_drop}"]

        # 3. Navigate to nearest item
        if state.items:
            target_pos, path = self._nearest_item_path(state)
            if path is not None:
                return path
            # If no path found, wait (shouldn't normally happen in valid maps)
            return ["wait"]

        # 4. Navigation-only goals (e.g. reach_beacon) — no item required
        #    Navigate directly to the goal zone.
        remaining_goals = [
            pos for pos, name in state.goals.items()
            if name not in state.completed_goals
        ]
        if remaining_goals:
            goal_pos = remaining_goals[0]
            path = self._bfs(state, (ac, ar), goal_pos)
            if path is not None:
                return path

        # Nothing left to do
        return ["wait"]

    def _nearest_item_path(self, state: WorldState):
        """
        BFS from agent position to find the nearest reachable item.
        Returns (item_pos, path_actions) or (None, None) if unreachable.
        """
        ac, ar = state.agent_pos
        best_pos = None
        best_path = None
        best_len = float("inf")

        for item_pos in state.items:
            path = self._bfs(state, (ac, ar), item_pos)
            if path is not None and len(path) < best_len:
                best_len = len(path)
                best_pos = item_pos
                best_path = path

        return best_pos, best_path

    def _bfs(
        self,
        state: WorldState,
        start: tuple,
        goal: tuple,
    ) -> list[str] | None:
        """
        BFS shortest path from start to goal on the grid.
        Returns list of action strings, or None if unreachable.
        Treats walls as impassable; items and goal zones are passable.
        """
        if start == goal:
            return []

        # (col, row) -> direction taken to reach it
        visited = {start: None}
        parent = {start: None}
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            col, row = pos

            for direction, (dc, dr) in DIRECTIONS.items():
                nc, nr = col + dc, row + dr
                npos = (nc, nr)

                if npos in visited:
                    continue
                if npos in state.walls:
                    continue
                if not (0 <= nc < state.width and 0 <= nr < state.height):
                    continue

                visited[npos] = direction
                parent[npos] = pos
                queue.append(npos)

                if npos == goal:
                    # Reconstruct path
                    return self._reconstruct_path(parent, visited, start, goal)

        return None  # unreachable

    @staticmethod
    def _reconstruct_path(
        parent: dict,
        visited: dict,
        start: tuple,
        goal: tuple,
    ) -> list[str]:
        """Walk back from goal to start using parent map; return action list."""
        actions = []
        pos = goal
        while pos != start:
            direction = visited[pos]
            actions.append(f"move_{direction}")
            pos = parent[pos]
        actions.reverse()
        return actions

    def reset(self):
        """Clear the planned path (useful between scenarios)."""
        self._path = []
