"""
llm_agent.py: Claude-backed LLM agent harness.

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
  - API errors (rate limits, overload) are retried with exponential backoff.
"""

from __future__ import annotations
import os
import re
import time
import anthropic
from world.grid import WorldState, get_observation, render_grid, get_visible_cells
from world.actions import get_action_space_description


HISTORY_WINDOW = 8    # keep last N (user+assistant) message pairs
MAX_TOKENS = 1024     # enough headroom for multi-step reasoning
MAX_RETRIES = 3       # API retry attempts on transient errors
RETRY_BASE_DELAY = 2  # seconds (doubles each retry)

VALID_ACTION_PREFIXES = [
    "move_north", "move_south", "move_east", "move_west",
    "pick_up", "drop ", "wait",
]

# Number of recent actions to track for repeat-action detection
RECENT_ACTIONS_WINDOW = 5
REPEAT_THRESHOLD = 3   # warn after this many consecutive identical actions


def _obs_to_text(obs: dict, grid_str: str, repeat_warning: str = None) -> str:
    """Convert structured observation dict to clear natural language.

    Parameters
    ----------
    obs : dict
        Structured observation returned by get_observation().
    grid_str : str
        ASCII grid string.
    repeat_warning : str, optional
        System warning injected when the agent is repeating the same action.
    """
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

    if obs.get("fog_of_war"):
        lines.append(
            f"\n[FOG OF WAR ACTIVE] Visibility radius: {obs['visible_radius']} cells "
            "(Manhattan distance). Cells shown as '?' are unexplored — you have not "
            "seen them yet."
        )

    if obs["at_item"]:
        lines.append(f"\nYou are standing ON a {obs['at_item']} — you can pick it up.")
    if obs["at_goal"]:
        lines.append(f"\nYou are standing ON the goal zone — drop your item(s) here to score.")

    if obs["known_items"]:
        label = "Visible item locations:" if obs.get("fog_of_war") else "Known item locations:"
        lines.append(f"\n{label}")
        for name, info in obs["known_items"].items():
            lines.append(f"  {name}: col {info['col']}, row {info['row']}")

    # Show previously discovered items not currently visible
    if obs.get("fog_of_war") and obs.get("discovered_items"):
        prev_only = {
            k: v for k, v in obs["discovered_items"].items()
            if k not in obs["known_items"]
        }
        if prev_only:
            lines.append("\nPreviously discovered items (may have moved or been picked up):")
            for name, info in prev_only.items():
                lines.append(f"  {name}: col {info['col']}, row {info['row']} (last seen)")

    if obs["known_goals"]:
        label = "Visible goal zones:" if obs.get("fog_of_war") else "Goal zone locations:"
        lines.append(f"\n{label}")
        for goal_name, info in obs["known_goals"].items():
            lines.append(f"  {goal_name}: col {info['col']}, row {info['row']}")

    if obs["nearest_item_distance"] is not None:
        lines.append(f"\nDistance to nearest item: {obs['nearest_item_distance']} steps (Manhattan)")

    if obs["completed_goals"]:
        lines.append(f"\nCompleted goals so far: {', '.join(obs['completed_goals'])}")

    fog_note = " (?=unexplored)" if obs.get("fog_of_war") else ""
    lines.append(
        f"\nWorld map (A=you, K=key, *=gem, C=crystal, G=goal, #=wall{fog_note}):\n{grid_str}"
    )

    if repeat_warning:
        lines.append(repeat_warning)

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
    FOG_MODE = False  # class-level default

    def __init__(self, scenario_goal: str, verbose: bool = True, fog_radius: int = None):
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

        # Fog of war
        self.fog_radius = fog_radius
        self.discovered: dict = {"items": {}, "goals": {}}

        # Repeat-action detection
        self.recent_actions: list[str] = []

    def choose_action(self, state: WorldState) -> str:
        """
        Given the current world state, ask Claude to choose an action.
        Returns an action string (e.g. 'move_east', 'pick_up').
        """
        # Build observation (with optional fog)
        if self.fog_radius is not None:
            obs = get_observation(state, fog_radius=self.fog_radius, discovered=self.discovered)
            visible = get_visible_cells(state, self.fog_radius)
            grid_str = render_grid(state, visible_cells=visible)
        else:
            obs = get_observation(state)
            grid_str = render_grid(state)

        # Build repeat-action warning if needed
        repeat_warning = self._build_repeat_warning()

        obs_text = _obs_to_text(obs, grid_str, repeat_warning=repeat_warning)

        self.messages.append({"role": "user", "content": obs_text})

        # Trim history to window
        trimmed = self.messages[-(HISTORY_WINDOW * 2):]

        raw_text = self._call_api_with_retry(trimmed)
        self.messages.append({"role": "assistant", "content": raw_text})

        if self.verbose:
            print(f"\n[Claude thinking]\n{raw_text}\n")

        action = _parse_action(raw_text)

        # Track recent actions for repeat detection
        self.recent_actions.append(action)
        if len(self.recent_actions) > RECENT_ACTIONS_WINDOW:
            self.recent_actions.pop(0)

        return action

    def _build_repeat_warning(self) -> str | None:
        """Return a warning string if the same action has been repeated >= REPEAT_THRESHOLD times."""
        if len(self.recent_actions) < REPEAT_THRESHOLD:
            return None
        last = self.recent_actions[-1]
        # Count how many trailing actions match the last one
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

    def _call_api_with_retry(self, messages: list[dict]) -> str:
        """
        Call the Anthropic API with exponential backoff retry on transient errors.
        Raises on non-retryable errors (auth, invalid request).
        """
        delay = RETRY_BASE_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=MAX_TOKENS,
                    system=self.system_prompt,
                    messages=messages,
                )
                return response.content[0].text.strip()

            except anthropic.RateLimitError as e:
                if attempt == MAX_RETRIES:
                    raise
                print(f"[WARN] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2

            except anthropic.APIStatusError as e:
                # Retry on 529 (overloaded) and 5xx server errors only
                if e.status_code in (529, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    print(f"[WARN] API error {e.status_code} (attempt {attempt}/{MAX_RETRIES}). "
                          f"Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

        # Should not reach here
        raise RuntimeError("API call failed after all retries.")

    def reset(self):
        self.messages = []
        self.discovered = {"items": {}, "goals": {}}
        self.recent_actions = []


def _parse_action(text: str) -> str:
    """
    Extract the action from the LLM response.
    Takes the LAST 'ACTION: <action>' occurrence to avoid misparses
    when Claude quotes the format in its reasoning.
    Falls back to keyword scan on the last line, then 'wait'.
    """
    # Find ALL ACTION: occurrences and take the last one
    matches = re.findall(r"ACTION:\s*(.+)", text, re.IGNORECASE)
    if matches:
        action = matches[-1
    