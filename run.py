"""
run.py: Entry point. Runs a scenario and logs results.

Usage:
    python run.py                             # default: delivery scenario, LLM agent
    python run.py --scenario multi            # multi-item delivery
    python run.py --max-steps 40              # override step limit
    python run.py --no-verbose                # suppress Claude's reasoning
    python run.py --log run_log.json          # save full log to file
    python run.py --fog                       # enable fog of war (default radius 4)
    python run.py --fog --fog-radius 3        # fog with custom radius
    python run.py --agent bfs                 # use deterministic BFS agent
    python run.py --agent compare             # run both agents and compare step counts
"""

import argparse
import json
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from world.grid import build_world, render_grid, get_observation
from world.actions import apply_action
from agent.llm_agent import LLMAgent
from agent.bfs_agent import BFSAgent


SCENARIO_GOALS = {
    "delivery": (
        "Pick up the KEY (K) and carry it to the GOAL zone (G) to complete the delivery task."
    ),
    "multi": (
        "Pick up both the GEM (*) and the CRYSTAL (C) and deliver them "
        "to the GOAL zone (G) to complete the task."
    ),
}

SCENARIO_COMPLETION = {
    "delivery": lambda state: "deliver_key" in state.completed_goals,
    "multi": lambda state: "deliver_gem_and_crystal" in state.completed_goals,
}


def run(
    scenario: str,
    max_steps: int,
    verbose: bool,
    log_path: str | None,
    fog: bool = False,
    fog_radius: int = 4,
    agent_type: str = "llm",
):
    print(f"\n{'='*60}")
    print(f"  LLM Agent World — Scenario: {scenario.upper()}")
    print(f"  Goal: {SCENARIO_GOALS[scenario]}")
    if fog and agent_type == "llm":
        print(f"  Fog of War: ENABLED (radius={fog_radius})")
    print(f"{'='*60}\n")

    state = build_world(scenario)

    if agent_type == "bfs":
        agent = BFSAgent()
    else:
        agent = LLMAgent(
            scenario_goal=SCENARIO_GOALS[scenario],
            verbose=verbose,
            fog_radius=fog_radius if fog else None,
        )

    print("Initial world:")
    print(render_grid(state))
    print()

    log_entries = []
    done = False

    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")
        print(render_grid(state))

        obs = get_observation(state)
        action = agent.choose_action(state)

        print(f"[Agent action] {action}")
        state, result = apply_action(state, action)
        print(f"[Result] {result}\n")

        log_entries.append({
            "step": step,
            "observation": obs,
            "action": action,
            "result": result,
            "grid": render_grid(state),
        })

        if SCENARIO_COMPLETION[scenario](state):
            print(f"\n✓ GOAL ACHIEVED in {step} steps!\n")
            done = True
            break

    if not done:
        print(f"\n✗ Goal not achieved within {max_steps} steps.\n")

    summary = {
        "scenario": scenario,
        "goal": SCENARIO_GOALS[scenario],
        "completed": done,
        "steps_taken": state.steps,
        "final_inventory": state.agent_inventory,
        "completed_goals": state.completed_goals,
        "final_grid": render_grid(state),
        "run_at": datetime.utcnow().isoformat(),
        "steps": log_entries,
    }

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Log saved to {log_path}")

    return summary


def run_comparison(scenario: str, max_steps: int, verbose: bool):
    """
    Run both LLM agent and BFS agent on the same scenario and print step counts side by side.
    """
    print(f"\n{'='*60}")
    print(f"  COMPARISON RUN — Scenario: {scenario.upper()}")
    print(f"{'='*60}\n")

    # --- BFS run ---
    print(">> Running BFS agent...")
    bfs_state = build_world(scenario)
    bfs_agent = BFSAgent()
    bfs_done = False
    for step in range(1, max_steps + 1):
        action = bfs_agent.choose_action(bfs_state)
        bfs_state, _ = apply_action(bfs_state, action)
        if SCENARIO_COMPLETION[scenario](bfs_state):
            bfs_done = True
            break
    bfs_steps = bfs_state.steps if bfs_done else None

    # --- LLM run ---
    print(">> Running LLM agent...")
    llm_state = build_world(scenario)
    llm_agent = LLMAgent(
        scenario_goal=SCENARIO_GOALS[scenario],
        verbose=verbose,
    )
    llm_done = False
    for step in range(1, max_steps + 1):
        action = llm_agent.choose_action(llm_state)
        llm_state, _ = apply_action(llm_state, action)
        if SCENARIO_COMPLETION[scenario](llm_state):
            llm_done = True
            break
    llm_steps = llm_state.steps if llm_done else None

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Comparison Results — {scenario.upper()}")
    print(f"{'='*60}")
    bfs_label = f"{bfs_steps} steps" if bfs_done else f"FAILED (>{max_steps} steps)"
    llm_label = f"{llm_steps} steps" if llm_done else f"FAILED (>{max_steps} steps)"
    print(f"  BFS Agent : {bfs_label}")
    print(f"  LLM Agent : {llm_label}")
    if bfs_done and llm_done:
        overhead = llm_steps - bfs_steps
        pct = (overhead / bfs_steps) * 100 if bfs_steps else 0
        sign = "+" if overhead >= 0 else ""
        print(f"  Overhead  : {sign}{overhead} steps ({sign}{pct:.0f}% vs BFS optimal)")
    print(f"{'='*60}\n")

    return {"bfs_steps": bfs_steps, "llm_steps": llm_steps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM agent in a virtual world.")
    parser.add_argument("--scenario", choices=["delivery", "multi"], default="delivery")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--no-verbose", action="store_true", help="Suppress Claude's reasoning output")
    parser.add_argument("--log", type=str, default="logs/run_log.json", help="Path to save JSON log")
    parser.add_argument("--fog", action="store_true", help="Enable fog of war")
    parser.add_argument("--fog-radius", type=int, default=4, help="Fog of war visibility radius (default 4)")
    parser.add_argument(
        "--agent",
        choices=["llm", "bfs", "compare"],
        default="llm",
        help="Agent to run: llm (default), bfs (deterministic), compare (both side by side)",
    )
    args = parser.parse_args()

    if args.agent == "compare":
        run_comparison(
            scenario=args.scenario,
            max_steps=args.max_steps,
            verbose=not args.no_verbose,
        )
    else:
        run(
            scenario=args.scenario,
            max_steps=args.max_steps,
            verbose=not args.no_verbose,
            log_path=args.log,
            fog=args.fog,
            fog_radius=args.fog_radius,
            agent_type=args.agent,
        )