"""
llm_agent.py — Claude-backed LLM agent harness.

The agent harness is responsible for:
  1. Translating world observations into a rich prompt
  2. Calling the LLM (Anthropic Claude)
  3. Parsing the LLM's response into a valid action string
  4. Returning the chosen action

Design notes:
  - Observations are serialised to structured natural language (not raw JSON)
    so the LLM can reason about them fluently.
  - The system prompt establishes the agent's identity, goal, and action space
    once at the start of the session.
  - A rolling message history is maintained so the LLM can reason over recent
    steps without hitting context limits (we keep last N turns).
  - Reasoning is extracted from <think>...</think> tags if the model provides it.
"""

from __future__ import annotations
import os
import re
import json
import anthropic
from world.grid import WorldState, get_observation, render_grid
from world.actions import get_action_space_description


HISTORY_WINDOW = 8   # keep last N (user+assistant) message pairs
MAX_TOKENS = 512


def _obs_to_text(obs: dict, grid_str: str) -> str:
    """Convert structured observation dict to clear natural language."""
    pos = obs["agent_position"]
    inv = obs["inventory"] or ["nothing"]
    surroundings = obs["surroundings"]
    lines = [
        f"=== Step {obs['steps_taken']} ===",
        f"Position: column {pos['col']}, row {pos['row']}",
        f"Inventory: {', '.join(inv)}",
        "",
        "Surroundings:",
    ]
    for direction, content in surroundings.items():
        lines.append(f"  {direction}: {content}")

    if obs["at_item"]:
        lines.append(f"\nYou are standing ON a {obs['at_item']} — you can pick it up.")
    if obs["at_goal"]:
        lines.append(f"\nYou are standing ON the goal zone — drop your item(s) here to score.")

    if obs["known_items"]:
        lines.append("\nKnown item locations:")
        for name, info in obs["known_items"].items():
            lines.append(f"  {name}: col {info['col']}, row {info['row']}")

    if obs["known_goals"]:
        lines.append("\nGoal zone locations:")
        for goal_name, info in obs["known_goals"].items():
            lines.append(f"  {goal_name}: col {info['col']}, row {info['row']}")

    if obs["completed_goals"]:
        lines.append(f"\nCompleted goals so far: {', '.join(obs['completed_goals'])}")

    lines.append(f"\nWorld map (A=you, K=key, *=gem, C=crystal, G=goal, #=wall):\n{grid_str}")
    return "\n".join(lines)


def _build_system_prompt(scenario_goal: str) -> str:
    return f"""You are an autonomous agent navigating a 2D grid world.

YOUR GOAL: {scenario_goal}

{get_action_space_description()}

Rules:
- You cannot walk through walls (#).
- You must be ON an item's cell to pick it up.
- You must be ON the goal zone (G) and have the required item to drop it and complete the goal.
- Think step by step. Plan a path to the item, then to the goal.

Response format:
Respond with your reasoning first (a few sentences), then end your message with:
ACTION: <your chosen action>

The ACTION line must be the very last line of your response.
Example: ACTION: move_east
"""


class LLMAgent:
    def __init__(self, scenario_goal: str, verbose: bool = True):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = _build_system_prompt(scenario_goal)
        self.messages: list[dict] = []
        self.verbose = verbose

    def choose_action(self, state: WorldState) -> str:
        """
        Given the current world state, ask Claude to choose an action.
        Returns an action string (e.g. 'move_east', 'pick_up').
        """
        obs = get_observation(state)
        grid_str = render_grid(state)
        obs_text = _obs_to_text(obs, grid_str)

        self.messages.append({"role": "user", "content": obs_text})

        # Trim history to window
        trimmed = self.messages[-(HISTORY_WINDOW * 2):]

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=MAX_TOKENS,
            system=self.system_prompt,
            messages=trimmed,
        )

        raw_text = response.content[0].text.strip()
        self.messages.append({"role": "assistant", "content": raw_text})

        if self.verbose:
            print(f"\n[Claude thinking]\n{raw_text}\n")

        action = _parse_action(raw_text)
        return action

    def reset(self):
        self.messages = []


def _parse_action(text: str) -> str:
    """
    Extract the action from the LLM response.
    Looks for 'ACTION: <action>' on the last line, falls back to scanning.
    """
    # Look for ACTION: line (case-insensitive)
    match = re.search(r"ACTION:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()

    # Fallback: check if last non-empty line is a known action keyword
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1].lower()
        known_starts = ["move_", "pick_up", "drop ", "wait"]
        for k in known_starts:
            if last.startswith(k):
                return last

    return "wait"  # safe fallback
