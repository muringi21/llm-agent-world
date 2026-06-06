# LLM Agent in a Virtual World

An autonomous LLM agent (Claude claude-opus-4-5) that perceives a 2D grid world, reasons about its state, and takes goal-directed actions to complete delivery tasks.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run a scenario

```bash
# Default: single-item delivery
python run.py

# Multi-item delivery
python run.py --scenario multi

# Suppress Claude's reasoning output (cleaner logs)
python run.py --no-verbose

# Save run log to a custom path
python run.py --log logs/my_run.json
```

---

## Scenarios

### `delivery` (default)
The agent must navigate an 8×8 grid, pick up a **KEY** (`K`), and deliver it to the **GOAL zone** (`G`). Interior walls create a simple maze that requires planning.

```
  0 1 2 3 4 5 6 7
0 # # # # # # # #
1 # A . # . . G #
2 # . . # . # . #
3 # . . # . # . #
4 # . . # . . . #
5 # . . # . K . #
6 # . . . . . . #
7 # # # # # # # #
```

### `multi`
The agent must collect both a **GEM** (`*`) and a **CRYSTAL** (`C`) from opposite sides of a 10×10 maze, then deliver both to the **GOAL zone** (`G`).

---

## Architecture

### World Representation

The world is a W×H grid of tiles. Each cell can be:
- Empty (`.`), Wall (`#`), Item (`K`, `*`, `C`), Goal zone (`G`), or the Agent (`A`)

The world state is tracked as a `WorldState` dataclass containing:
- Agent position and inventory
- Item and goal positions
- Step count, completed goals, and action history

### Observation Format

At each step, the agent receives a structured observation containing:
- Current position and inventory
- Adjacent cell contents (north/south/east/west)
- All known item and goal positions (the agent has full map knowledge)
- Steps taken and goals completed so far

The observation is converted to natural language before being passed to the LLM, this gives Claude the richest possible context without the overhead of parsing raw JSON mid-reasoning.

### Action Space

```
move_north / move_south / move_east / move_west
pick_up
drop <item_name>
wait
```

Actions are validated before application. Invalid moves (into walls, out-of-bounds) return an error message and the state is unchanged.

### Agent Harness

The `LLMAgent` class in `agent/llm_agent.py` manages:
1. **System prompt** - given once, establishes goal, action space, and response format
2. **Rolling message history** - last 8 user/assistant pairs are kept so Claude reasons over recent steps without hitting context limits
3. **Observation rendering** - `_obs_to_text()` converts structured state to clear natural language + ASCII grid
4. **Action parsing** - extracts `ACTION: <action>` from Claude's response; falls back to keyword scan if needed

### Why this design?

**Observation representation** is the most important decision. The agent needs to know:
- Where it is now
- What's reachable nearby (surroundings)
- Where the targets are (full map knowledge, no fog-of-war)

I chose to give the agent full positional knowledge of items and goals. This removes the need for exploration behaviour and lets us focus on demonstrating clean goal-directed planning, which is the core of the challenge.

The ASCII grid render is included alongside structured text so Claude can visually reason about spatial relationships. This dual representation (structured + visual) consistently outperforms either alone.

**Message history** lets Claude implicitly learn from failed moves (hitting a wall, taking a detour) without needing explicit reflection logic. The last-N-turns window keeps context bounded.

---

## Example Output

```
=== Step 1 ===
Position: column 1, row 1
Inventory: nothing

Surroundings:
  north: wall
  south: empty
  east: empty
  west: wall

Known item locations:
  key: col 5, row 5

Goal zone locations:
  deliver_key: col 6, row 1

[Claude thinking]
I'm at (1,1) with nothing in my inventory. The key is at (5,5) and
the goal is at (6,1). I need to go south and east to reach the key,
but there's a wall segment at column 3 rows 2-5 blocking the direct path.
I should move east first, then navigate around the wall.

ACTION: move_east

[Result] Moved east to (2, 1).

--- Step 7 ---
...
[Agent picks up key at (5,5)]
...
--- Step 14 ---
[Agent drops key at goal (6,1)]
✓ GOAL ACHIEVED in 12 steps!
```

Full JSON logs (including all observations, actions, and results) are saved to `logs/run_log.json`.

---

## Project Structure

```
llm-agent-world/
├── agent/
│   └── llm_agent.py      # LLM harness - prompting, history, action parsing
├── world/
│   ├── grid.py           # World state, rendering, observation builder
│   └── actions.py        # Action space and state transition logic
├── logs/                 # Run logs (JSON)
├── run.py                # Entry point and scenario runner
├── requirements.txt
└── .env.example
```

---

## Design Choices

| Decision | Rationale |
|---|---|
| Text-based 2D grid | Simple to render, easy to include in LLM prompts, no UI dependencies |
| Full map knowledge | Focuses the demo on planning/execution, not exploration |
| Dual observation (structured + ASCII) | Claude reasons better with both spatial and structured representations |
| Rolling message window | Bounded context cost while preserving recent step memory |
| `ACTION:` structured output | Reliable parsing without JSON overhead |
| Separate world/agent modules | Clean separation between environment logic and LLM harness |

---

## Extending

- Add **fog-of-war** - limit observation to a radius around the agent
- Add **multi-agent** - two agents collaborating on the same world
- Add **a FastAPI server** exposing the world as a REST API for external clients
- Swap the LLM - the `LLMAgent` class is model-agnostic; replace the `anthropic` call with any provider
