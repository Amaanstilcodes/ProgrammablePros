"""Batch Runner Tool — Genesis 2.0 Track EVA 02.

Executes a series of game sessions, calculates median scores (as used for Stage 2 judging),
and verifies compliance metrics across the batch.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from orchestrator_compliance import run_orchestrated_game


def run_batch(game: str, count: int, mock: bool = True, max_steps: int = 50) -> list[int]:
    print(f"[*] Starting batch of {count} sessions for game: {game} (Mock={mock})")
    scores: list[int] = []

    for i in range(1, count + 1):
        print(f"\n--- Batch Run {i}/{count} ---")
        result = run_orchestrated_game(
            game=game,
            is_mock=mock,
            max_steps_override=max_steps,
        )
        # Load final score
        final_state = result.get("final_state_path")
        score = 0
        if final_state:
            import json
            with open(final_state, "r", encoding="utf-8") as f:
                data = json.load(f)
                score = int(data.get("score", 0))
        scores.append(score)

    print("\n================================================================================")
    print(f"[*] Batch Run Complete — Total Runs: {count}")
    print(f"    Scores: {scores}")
    print(f"    Median Score: {statistics.median(scores):.1f}")
    print(f"    Max Score: {max(scores)}")
    print("================================================================================")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch evaluation sessions")
    parser.add_argument("--game", choices=["snake", "2048"], default="snake")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--mock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    run_batch(game=args.game, count=args.count, mock=args.mock, max_steps=args.max_steps)
