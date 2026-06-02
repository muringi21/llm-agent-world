"""
run.py: Entry point. Runs a scenario and logs results.

Usage:
    python run.py                    # default: delivery scenario
    python run.py --scenario multi   # multi-item delivery
    python run.py --max-steps 40     # override step limit
    python run.py --no-verbose       # suppress Claude's reasoning
    python run.py --log run_log.json # save full log to file
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


def run(scenario: str, max_steps: int, verbose: bool, log_path: str | None):
    print(f"\n{'='*60}")
    print(f"  LLM Agent World — Scenario: {scenario.upper()}")
    print(f"  Goal: {SCENARIO_GOALS[scenario]}")
    print(f"{'='*60}\n")

    state = build_world(scenario)
    agent = LLMAgent(scenario_goal=SCENARIO_GOALS[scenario], verbose=verbose)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM agent in a virtual world.")
    parser.add_argument("--scenario", choices=["delivery", "multi"], default="delivery")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--no-verbose", action="store_true", help="Suppress Claude's reasoning output")
    parser.add_argument("--log", type=str, default="logs/run_log.json", help="Path to save JSON log")
    args = parser.parse_args()

    run(
        scenario=args.scenario,
        max_steps=args.max_steps,
        verbose=not args.no_verbose,
        log_path=args.log,
    )
