# LLM Agent in a Virtual World

The agent doesn't just act , it *perceives*. A dual-representation observation layer (structured text + ASCII grid) gives the LLM simultaneous symbolic and spatial access to world state, letting it plan routes, avoid walls, and sequence multi-item deliveries without hand-holding. The harness is intentionally thin: bounded message history, a single structured output token, and clean world/agent separation keep the system auditable and extensible.

```bash
python run.py                         # delivery scenario (default)
python run.py --scenario multi        # multi-item collection + delivery
python run.py --agent bfs             # deterministic BFS baseline
python run.py --fog --fog-radius 2    # partial observability
```

---

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         WORLD (grid.py)                         │
│  W×H tile grid  ·  WorldState dataclass  ·  fog-of-war radius   │
└────────────────────────────┬────────────────────────────────────┘
                             │  raw state
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OBSERVATION BUILDER                           │
│  structured text (position, inventory, adjacency, goals)        │
│  + ASCII grid render (agent, items, walls, fog tiles)           │
│  + repeat-action nudge (injected if stuck loop detected)        │
└────────────────────────────┬────────────────────────────────────┘
                             │  natural-language observation
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM HARNESS (llm_agent.py)                    │
│  system prompt  ·  rolling 8-turn history  ·  action parser     │
│  Claude claude-opus-4-5 (swappable — any provider works)        │
└────────────────────────────┬────────────────────────────────────┘
                             │  ACTION: <token>
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ACTION LAYER (actions.py)                     │
│  validation  ·  state transition  ·  error feedback             │
└────────────────────────────┬────────────────────────────────────┘
                             │  updated WorldState
                             ▼
                        WORLD (loop)

```                  

**Fog-of-war** masks tiles outside a configurable Manhattan-distance radius; a distance hint to the nearest item replaces full positional knowledge, forcing the LLM to explore rather than teleport to the answer.

**BFS baseline** provides a deterministic lower bound on step count for any scenario, useful for scoring LLM efficiency without needing to run multiple API calls.

**Repeat-action detection** watches the last N actions in `WorldState.action_history`. If the agent is oscillating (A→B→A→B), a plain-English nudge is injected into the next observation: *"You have taken this action repeatedly without progress; consider a different approach."* No reward shaping, no hidden state; just honest feedback in the observation channel.

---

## Design Decisions

| Decision | What it does | Evaluation criterion |
|---|---|---|
| **Dual observation: structured text + ASCII grid** | Structured text gives the LLM exact symbolic facts (position, inventory, goal coords). The ASCII grid encodes topology: walls, corridors, relative distances, in a form Claude has seen millions of times in training. Together they consistently outperform either alone on spatial tasks. | Observation thoughtfulness |
| **Fog-of-war as a first-class feature** | Flipping `--fog` turns a planning problem into an exploration problem. The same harness, prompt, and parser handle both. | Harness design, creativity |
| **BFS baseline agent** | Deterministic optimal pathing, a concrete benchmark, not a vibe check. | Harness design, simplicity |
| **Repeat-action detection via observation injection** | Stuck-loop detection through the normal observation channel, not a separate control path. | Creativity, simplicity |
| **Rolling 8-turn message history** | Bounded context cost. The history *is* the memory. | Harness design, simplicity |
| **`ACTION:` structured output token** | One token, reliable regex parse, no JSON overhead. Never fails silently. | Simplicity |
| **Separate world / agent modules** | `world/` has zero knowledge of LLMs; `agent/` has zero knowledge of tile types. | Harness design |

---

## Observation Representation

The observation the LLM receives at each step is the highest-leverage surface in the system. Getting it wrong; too sparse, too verbose, wrong format, causes more task failures than any prompt engineering issue.

Two representations are generated and concatenated every step:

### 1. Structured text
Position: column 2, row 3
Inventory: key
Surroundings:
  north: wall
  south: empty
  east: empty
  west: wall
Known item locations:
  gem: col 7, row 2
Goal zone locations:
  deliver_gem: col 1, row 8
Steps taken: 9  |  Goals completed: 0/1

This gives the LLM **exact symbolic facts** with zero ambiguity. Structured text is fast to generate, deterministic, and trivial to extend with new fields.

### 2. ASCII grid
```
  0 1 2 3 4 5 6 7 8 9
0 # # # # # # # # # #
1 # . . # . . . * . #
2 # . . # . # . # . #
3 # . A # . # . # . #
4 # . . . . # . . . #
5 # . . # . . . # . #
6 # . . # . # . # . #
7 # . . . . . . . . #
8 # G . . . . . . . #
9 # # # # # # # # # #
```
This gives the LLM **spatial topology** in a format it has encountered extensively in pre-training. Claude can read this grid and immediately see wall clusters, corridors, and relative distances that would require multi-step inference from structured text alone.

**Why both?** Structured text answers *what*. The ASCII grid answers *where*. The dual representation reduces wasted moves and wall-collision loops compared to either format alone.

**Fog-of-war variant:** When `--fog` is enabled, tiles outside the radius render as `?`, and structured text replaces exact coordinates with Manhattan-distance hints.

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

The agent must collect both a **GEM** (`*`) and a **CRYSTAL** (`C`) from opposite sides of a 10×10 maze, then deliver both to the **GOAL zone** (`G`). Sequencing matters.

---

## Action Space

move_north / move_south / move_east / move_west
pick_up
drop <item_name>
wait

Actions are validated before application. Invalid moves return a plain-English error that appears in the next observation — no special error-handling logic needed in the harness.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt


### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run a scenario

```bash
python run.py
python run.py --scenario multi
python run.py --fog --fog-radius 2
python run.py --agent bfs
python run.py --no-verbose
python run.py --log logs/my_run.json
```
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
Full JSON logs saved to `logs/run_log.json`.

---

## Project Structure
```
llm-agent-world/
├── agent/
│   ├── llm_agent.py      # LLM harness- prompting, history, action parsing
│   └── bfs_agent.py      # Deterministic BFS baseline agent
├── world/
│   ├── grid.py           # World state, rendering, observation builder
│   └── actions.py        # Action space and state transition logic
├── tests/
│   └── test_agent.py     # 34 unit tests (no API key required)
├── assets/
│   └── test_results.png  # Test suite output
├── logs/                 # Run logs (JSON)
├── run.py                # Entry point and scenario runner
├── requirements.txt
└── .env.example
```
---

## Testing

34 tests covering action parsing, world logic, fog-of-war, BFS agent, and repeat-action detection. No API key required.

```bash
python3 -m pytest tests/ -v
```

![34 tests passing](assets/test_results.png)

---

## Extending

**Already implemented:**
- **Fog-of-war** — configurable radius, Manhattan-distance hint to nearest item
- **BFS baseline agent** — deterministic pathfinding for benchmarking
- **Repeat-action detection** — penalises stuck loops via observation nudge

**Future ideas:**
- Multi-agent collaboration
- FastAPI server (`POST /step` returns next observation)
- Procedural map generation
- Swap the LLM — `LLMAgent` is model-agnostic