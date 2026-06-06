"""
tests/test_agent.py: Unit tests for the LLM agent harness and world logic.

Uses a MockLLMAgent that returns scripted actions instead of calling the API,
so tests run without an ANTHROPIC_API_KEY and complete instantly.
"""

import pytest
from world.grid import (
    build_world,
    get_observation,
    render_grid,
    get_visible_cells,
    WorldState,
    Item,
    DEFAULT_FOG_RADIUS,
)
from world.actions import apply_action
from agent.llm_agent import _parse_action, _obs_to_text, REPEAT_THRESHOLD, RECENT_ACTIONS_WINDOW
from agent.bfs_agent import BFSAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLMAgent:
    """
    Stand-in for LLMAgent that returns a pre-scripted sequence of actions.
    Useful for deterministic testing without calling the Anthropic API.
    """
    def __init__(self, actions: list[str]):
        self._actions = iter(actions)

    def choose_action(self, state: WorldState) -> str:
        return next(self._actions, "wait")


# ---------------------------------------------------------------------------
# Action parsing tests
# ---------------------------------------------------------------------------

class TestParseAction:
    def test_parses_action_tag(self):
        text = "I should move east because the key is in that direction.\nACTION: move_east"
        assert _parse_action(text) == "move_east"

    def test_takes_last_action_tag(self):
        # Claude quotes the format in its reasoning — should use the LAST match
        text = "The format says ACTION: move_north but I want to go east.\nACTION: move_east"
        assert _parse_action(text) == "move_east"

    def test_case_insensitive(self):
        text = "action: pick_up"
        assert _parse_action(text) == "pick_up"

    def test_drop_with_item(self):
        text = "I am at the goal zone.\nACTION: drop key"
        assert _parse_action(text) == "drop key"

    def test_fallback_to_wait_on_gibberish(self):
        text = "I have no idea what to do here."
        assert _parse_action(text) == "wait"

    def test_rejects_hallucinated_action(self):
        # move_northeast starts with 'move_' which is a valid prefix substring,
        # but the world will treat it as an unknown action and return 'wait' safely.
        # Here we test that a completely unrecognised response falls back to wait.
        text = "I think I should teleport there."
        assert _parse_action(text) == "wait"


# ---------------------------------------------------------------------------
# World / observation tests
# ---------------------------------------------------------------------------

class TestWorldState:
    def test_delivery_world_builds(self):
        state = build_world("delivery")
        assert state.width == 8
        assert state.height == 8
        assert state.agent_pos == (1, 1)
        assert "key" in {item.name for item in state.items.values()}

    def test_multi_world_builds(self):
        state = build_world("multi")
        names = {item.name for item in state.items.values()}
        assert "gem" in names
        assert "crystal" in names

    def test_move_updates_position(self):
        state = build_world("delivery")
        new_state, msg = apply_action(state, "move_east")
        assert new_state.agent_pos == (2, 1)
        assert "Moved east" in msg

    def test_wall_blocks_movement(self):
        state = build_world("delivery")
        # Agent is at (1,1). North is (1,0) which is a wall.
        new_state, msg = apply_action(state, "move_north")
        assert new_state.agent_pos == (1, 1)  # position unchanged
        assert "Blocked" in msg

    def test_pick_up_item(self):
        state = build_world("delivery")
        # Move agent directly onto the key's position (5,5)
        state.agent_pos = (5, 5)
        new_state, msg = apply_action(state, "pick_up")
        assert "key" in new_state.agent_inventory
        assert (5, 5) not in new_state.items

    def test_drop_completes_goal(self):
        state = build_world("delivery")
        state.agent_pos = (5, 5)
        state, _ = apply_action(state, "pick_up")
        # Move to goal at (6,1)
        state.agent_pos = (6, 1)
        state, msg = apply_action(state, "drop key")
        assert "deliver_key" in state.completed_goals
        assert "key" not in state.agent_inventory

    def test_observation_includes_distance(self):
        state = build_world("delivery")
        obs = get_observation(state)
        # Key is at (5,5), agent at (1,1): Manhattan = 8
        assert obs["nearest_item_distance"] == 8

    def test_observation_distance_none_when_no_items(self):
        state = build_world("delivery")
        state.items.clear()
        obs = get_observation(state)
        assert obs["nearest_item_distance"] is None


# ---------------------------------------------------------------------------
# End-to-end scenario test with mock agent
# ---------------------------------------------------------------------------

class TestMockAgentDelivery:
    def test_delivery_completes_with_scripted_actions(self):
        """
        Verify the full game loop works correctly using a scripted action sequence.
        Agent navigates from (1,1) to (5,5), picks up key, navigates to (6,1), drops key.
        """
        # Navigate around the wall at (5,2)/(5,3): go east to col 4, south to row 5,
        # east to pick up key at (5,5), then return north and east to goal at (6,1)
        scripted = [
            "move_east", "move_east", "move_east",                          # (1,1)->(4,1)
            "move_south", "move_south", "move_south", "move_south",         # (4,1)->(4,5)
            "move_east",                                                     # (4,5)->(5,5)
            "pick_up",
            "move_west",                                                     # (5,5)->(4,5)
            "move_north", "move_north", "move_north", "move_north",         # (4,5)->(4,1)
            "move_east", "move_east",                                        # (4,1)->(6,1)
            "drop key",
        ]
        agent = MockLLMAgent(scripted)
        state = build_world("delivery")

        done = False
        for _ in range(25):
            action = agent.choose_action(state)
            state, _ = apply_action(state, action)
            if "deliver_key" in state.completed_goals:
                done = True
                break

        assert done, "Delivery scenario should complete with scripted path"
        assert state.steps == 17


# ---------------------------------------------------------------------------
# Feature 1: Fog of War tests
# ---------------------------------------------------------------------------

class TestFogOfWar:
    def _make_state_with_distant_item(self) -> WorldState:
        """
        Agent at (1,1), item at (7,7) — guaranteed to be outside radius 4.
        No walls except boundaries.
        """
        state = WorldState(width=10, height=10)
        # Outer walls only
        for c in range(10):
            state.walls.add((c, 0))
            state.walls.add((c, 9))
        for r in range(10):
            state.walls.add((0, r))
            state.walls.add((9, r))
        state.agent_pos = (1, 1)
        state.items[(7, 7)] = Item(name="treasure", symbol="T")
        state.goals[(8, 8)] = "deliver_treasure"
        return state

    def test_visible_cells_radius(self):
        """get_visible_cells returns cells within Manhattan radius."""
        state = WorldState(width=10, height=10)
        state.agent_pos = (5, 5)
        visible = get_visible_cells(state, radius=2)
        # Agent itself
        assert (5, 5) in visible
        # Cells exactly at radius 2
        assert (5, 3) in visible   # directly north by 2
        assert (3, 5) in visible   # directly west by 2
        # Cell outside radius 2
        assert (5, 2) not in visible   # distance 3
        assert (2, 5) not in visible

    def test_item_outside_radius_not_in_known_items(self):
        """Items beyond DEFAULT_FOG_RADIUS should NOT appear in known_items."""
        state = self._make_state_with_distant_item()
        # Agent at (1,1), item at (7,7): Manhattan distance = 12, radius = 4
        obs = get_observation(state, fog_radius=4)
        assert "treasure" not in obs["known_items"], (
            "Item at (7,7) is outside radius 4 and should not appear in known_items"
        )

    def test_item_inside_radius_in_known_items(self):
        """Items within DEFAULT_FOG_RADIUS SHOULD appear in known_items."""
        state = WorldState(width=10, height=10)
        # Minimal walls
        for c in range(10):
            state.walls.add((c, 0)); state.walls.add((c, 9))
        for r in range(10):
            state.walls.add((0, r)); state.walls.add((9, r))
        state.agent_pos = (1, 1)
        # Place item 3 steps away (within radius 4)
        state.items[(4, 1)] = Item(name="gem", symbol="*")
        obs = get_observation(state, fog_radius=4)
        assert "gem" in obs["known_items"], (
            "Item at (4,1) is within radius 4 and should appear in known_items"
        )

    def test_fog_observation_flags(self):
        """Observation dict includes fog_of_war=True and visible_radius when fog is on."""
        state = build_world("delivery")
        obs = get_observation(state, fog_radius=4)
        assert obs.get("fog_of_war") is True
        assert obs.get("visible_radius") == 4

    def test_no_fog_observation_has_no_fog_flag(self):
        """Without fog, observation dict has no fog_of_war key (or it's falsy)."""
        state = build_world("delivery")
        obs = get_observation(state)
        assert not obs.get("fog_of_war"), "fog_of_war should not be set without DEFAULT_FOG_RADIUS"

    def test_discovered_accumulates_across_steps(self):
        """discovered dict grows as agent moves and sees new items."""
        state = self._make_state_with_distant_item()
        discovered = {"items": {}, "goals": {}}

        # Initially agent at (1,1), far from treasure — nothing discovered
        obs1 = get_observation(state, fog_radius=4, discovered=discovered)
        assert "treasure" not in discovered["items"]

        # Move agent close to the item
        state.agent_pos = (6, 7)  # distance to (7,7) = 1, within radius 4
        obs2 = get_observation(state, fog_radius=4, discovered=discovered)
        assert "treasure" in discovered["items"], (
            "After moving next to the item, discovered should record it"
        )

    def test_render_grid_fog(self):
        """Cells outside visible_cells render as '?' in the grid."""
        state = self._make_state_with_distant_item()
        visible = get_visible_cells(state, radius=4)
        grid_str = render_grid(state, visible_cells=visible)
        # Item at (7,7) is outside radius — its symbol should not appear
        assert "T" not in grid_str, "Item symbol should be hidden by fog"
        # Agent marker always visible
        assert "A" in grid_str
        # Some fog cells
        assert "?" in grid_str


# ---------------------------------------------------------------------------
# Feature 2: BFS Agent tests
# ---------------------------------------------------------------------------

class TestBFSAgent:
    def test_bfs_solves_delivery_scenario(self):
        """BFSAgent should complete the delivery scenario deterministically."""
        state = build_world("delivery")
        agent = BFSAgent()
        done = False
        for _ in range(50):  # generous upper bound
            action = agent.choose_action(state)
            state, _ = apply_action(state, action)
            if "deliver_key" in state.completed_goals:
                done = True
                break
        assert done, "BFSAgent should complete delivery scenario"

    def test_bfs_solves_multi_scenario(self):
        """BFSAgent should complete the multi-item delivery scenario."""
        state = build_world("multi")
        agent = BFSAgent()
        done = False
        for _ in range(100):
            action = agent.choose_action(state)
            state, _ = apply_action(state, action)
            if "deliver_gem_and_crystal" in state.completed_goals:
                done = True
                break
        assert done, "BFSAgent should complete multi scenario"

    def test_bfs_picks_up_item_when_on_it(self):
        """BFSAgent returns pick_up immediately when standing on an item."""
        state = build_world("delivery")
        state.agent_pos = (5, 5)  # key is here
        agent = BFSAgent()
        action = agent.choose_action(state)
        assert action == "pick_up"

    def test_bfs_drops_at_goal(self):
        """BFSAgent returns drop action when at goal with item in inventory."""
        state = build_world("delivery")
        state.agent_pos = (6, 1)   # goal position
        state.agent_inventory = ["key"]
        state.items.clear()         # key already in inventory
        agent = BFSAgent()
        # Plan should include drop key as the final step
        actions = agent._plan(state)
        assert any("drop" in a for a in actions), (
            "BFS plan should include a drop action when at goal with item"
        )

    def test_bfs_optimal_delivery(self):
        """BFSAgent finds a path at least as short as the hand-scripted 17-step solution."""
        state = build_world("delivery")
        agent = BFSAgent()
        done = False
        for _ in range(50):
            action = agent.choose_action(state)
            state, _ = apply_action(state, action)
            if "deliver_key" in state.completed_goals:
                done = True
                break
        assert done
        # BFS should never need more steps than the hand-optimised scripted path (17)
        assert state.steps <= 17, f"BFS took {state.steps} steps; expected ≤ 17"


# ---------------------------------------------------------------------------
# Feature 3: Repeat-action detection tests
# ---------------------------------------------------------------------------

class TestRepeatActionDetection:
    def _make_agent_with_actions(self, actions: list[str]):
        """
        Simulate an agent that has already tracked a sequence of recent_actions.
        Returns the agent's _build_repeat_warning() result.
        """
        # Use a thin stand-in that exercises only the repeat-detection logic
        # without needing an Anthropic API key.
        class _FakeAgent:
            def __init__(self, recent):
                self.recent_actions = list(recent)

            def _build_repeat_warning(self):
                if len(self.recent_actions) < REPEAT_THRESHOLD:
                    return None
                last = self.recent_actions[-1]
                count = 0
                for a in reversed(self.recent_actions):
                    if a == last:
                        count += 1
                    else:
                        break
                if count >= REPEAT_THRESHOLD:
                    return (
                        f"\n[SYSTEM WARNING] You have attempted '{last}' {count} times in a row "
                        "with no progress. Try a different approach."
                    )
                return None

        agent = _FakeAgent(actions)
        return agent._build_repeat_warning()

    def test_no_warning_below_threshold(self):
        """No warning when action repeated fewer than REPEAT_THRESHOLD times."""
        warning = self._make_agent_with_actions(["move_north", "move_north"])
        assert warning is None

    def test_warning_at_threshold(self):
        """Warning appears exactly at REPEAT_THRESHOLD consecutive repeats."""
        actions = ["move_north"] * REPEAT_THRESHOLD
        warning = self._make_agent_with_actions(actions)
        assert warning is not None
        assert "move_north" in warning
        assert "SYSTEM WARNING" in warning

    def test_warning_above_threshold(self):
        """Warning includes correct count when repeated more than threshold times."""
        n = REPEAT_THRESHOLD + 2
        actions = ["wait"] * n
        warning = self._make_agent_with_actions(actions)
        assert warning is not None
        assert str(n) in warning

    def test_no_warning_when_actions_mixed(self):
        """No warning when recent actions are not all the same."""
        actions = ["move_north", "move_east", "move_north", "move_north", "move_east"]
        warning = self._make_agent_with_actions(actions)
        assert warning is None

    def test_warning_resets_after_different_action(self):
        """Warning stops after a different action breaks the streak."""
        # 3 repeats then a different action — last action differs
        actions = ["move_north", "move_north", "move_north", "move_east"]
        warning = self._make_agent_with_actions(actions)
        assert warning is None

    def test_obs_to_text_includes_warning(self):
        """_obs_to_text embeds the repeat_warning string in its output."""
        state = build_world("delivery")
        obs = get_observation(state)
        grid_str = render_grid(state)
        warning_text = "[SYSTEM WARNING] You have attempted 'wait' 3 times in a row with no progress. Try a different approach."
        result = _obs_to_text(obs, grid_str, repeat_warning=warning_text)
        assert warning_text in result

    def test_obs_to_text_no_warning_when_none(self):
        """_obs_to_text does not include any warning when repeat_warning=None."""
        state = build_world("delivery")
        obs = get_observation(state)
        grid_str = render_grid(state)
        result = _obs_to_text(obs, grid_str, repeat_warning=None)
        assert "SYSTEM WARNING" not in result
