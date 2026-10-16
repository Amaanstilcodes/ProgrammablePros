"""run_snake_live.py — Production Live Runner for Snake on TopGameAI Arena.

HACKATHON DAY DEPLOYMENT — ZERO MOCKS ALLOWED:
1. Connects to the REAL TopGameAI Arena API: https://topgameai.com/api/arena
2. Uses the REAL Laya model: convaiinnovations/laya
3. Autonomous decision loop:
   Compression -> Subgoal Arbitration -> Real Laya Inference -> Multi-term EV -> Anti-Reversal Safety -> Arena Move Dispatch -> Hash-Chained JSONL Logging
4. Logs each move with SHA-256 cryptographic hash chain adhering strictly to Genesis 2.0 competition rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add repo root to sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compliance_layer import ComplianceLayer
from game_clients import BaseGameClient, GameAPIError, MoveResult, SnakeClient
from laya_agent import LayaAgent
from official_logger import OfficialLogger
from snake_brain import (
    SnakeCompressor,
    SnakeEVCalculator,
    SnakeHypothesisBuilder,
    SnakeSubgoalManager,
    parse_snake_state,
)

LEGAL_ACTIONS: list[str] = ["up", "down", "left", "right"]
MAX_STEPS_SNAKE: int = 300


def run_live_snake_session(
    agent_name: str = "ProgrammablePros",
    max_steps: int = MAX_STEPS_SNAKE,
    log_dir: str = "logs/snake",
) -> dict[str, Any]:
    """Execute a real live Snake arena game with zero mocks."""
    print("\n" + "=" * 80)
    print("🚀 HACKATHON DAY: LIVE TOPGAMEAI ARENA SNAKE COMPETITION RUNNER")
    print("   Team: ProgrammablePros | Track: EVA 02 (Genesis 2.0)")
    print("   Mode: REAL LAYA MODEL + REAL TOPGAMEAI API (ZERO MOCKS ALLOWED)")
    print("=" * 80 + "\n")

    # 1. Initialize Real Laya Agent
    # Fails loudly if model weights cannot be loaded
    laya_engine = LayaAgent(model_id="convaiinnovations/laya")

    # 2. Initialize Real TopGameAI Arena Snake Client
    # Connects directly to https://topgameai.com/api/arena
    client = SnakeClient(agent_name=agent_name)

    # 3. Initialize Decision & Perception Pipeline
    compressor = SnakeCompressor()
    subgoal_mgr = SnakeSubgoalManager()
    hypo_builder = SnakeHypothesisBuilder()
    ev_calc = SnakeEVCalculator()
    compliance = ComplianceLayer()

    # 4. Initialize Official SHA-256 Hash-Chained Logger
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = OfficialLogger(log_dir=log_dir)

    # 5. Start Real Arena Session
    print("\n[*] Contacting TopGameAI Arena: POST /sessions ...")
    try:
        session_id, state_before = client.start_game(agent_name=agent_name)
    except Exception as exc:
        raise RuntimeError(f"FATAL: Failed to create TopGameAI Snake arena session: {exc}") from exc

    log_path = logger.start_session(session_id=session_id, game_name="snake")
    print(f"[+] Real Session Created: {session_id}")
    print(f"[+] Logging to Official JSONL: {log_path.resolve()}\n")

    # Initial State Overview
    parsed_init = parse_snake_state(state_before)
    print(f"[*] Initial Board: {parsed_init['width']}x{parsed_init['height']}")
    print(f"[*] Snake Head: {parsed_init['head']} | Length: {len(parsed_init['snake'])}")
    print(f"[*] Food Position: {parsed_init['food']}")
    print(f"[*] Travel Direction: {parsed_init['direction'].upper()}")
    print("-" * 80 + "\n")

    game_over = False
    won = False
    step_ordinal = 0
    total_turns = 0
    fallback_count = 0

    try:
        while not game_over and step_ordinal < max_steps:
            turn_start = time.perf_counter()
            total_turns += 1

            # A. State Perception & Compression (<= 200 chars)
            compressed_ctx = compressor.compress(state_before)

            # B. Tactical Subgoal Arbitration
            subgoal = subgoal_mgr.get_current_subgoal(state_before)

            # C. Hypothesis Strings for Official Log Compliance
            question_strings = hypo_builder.build_hypothesis_strings(state_before, compressed_ctx)

            # D. Query the REAL Laya Model (ZERO MOCKS)
            decide_start = time.perf_counter()
            action_probs = laya_engine.decide(
                state=state_before,
                possible_actions=LEGAL_ACTIONS,
                compressed_state=compressed_ctx,
            )

            # E. Multi-Term Expected Value Calculation (Includes Food-Seeking Manhattan Distance Bonus)
            proposed_action, ev_breakdown = ev_calc.calculate(
                state=state_before,
                raw_probs=action_probs,
                subgoal=subgoal,
            )

            # F. Compliance Anti-Reversal Safety Interceptor
            final_action = proposed_action
            source = "laya"
            fallback = compliance.check_snake_safety_fallback(state_before, proposed_action)
            if fallback is not None:
                final_action = fallback
                source = "fallback"
                fallback_count += 1
                compressor.record_failure(proposed_action)
                print(f"[!] SAFETY FALLBACK: Reversal '{proposed_action}' blocked -> Chose '{final_action}'")
            else:
                compressor.clear_failure()

            # Record action in EV history for repeat-penalty tracking
            ev_calc.record_move(final_action)

            decide_ms = (time.perf_counter() - decide_start) * 1000.0

            # G. Normalize Primary Probabilities for the Official Log (PDF §3.2)
            normalized_probs = compliance.normalize_laya_probabilities(action_probs)

            # H. Build Canonical Decision Block
            decision_block = compliance.build_decision_block(
                questions=question_strings,
                normalized_probs=normalized_probs,
                chosen_action=final_action,
                source=source,
            )

            # I. Dispatch Move to the REAL TopGameAI Arena
            api_start = time.perf_counter()
            move_res: MoveResult = client.send_move(
                session_id=session_id,
                action=final_action,
                state_before=state_before,
            )
            api_ms = (time.perf_counter() - api_start) * 1000.0
            turn_ms = (time.perf_counter() - turn_start) * 1000.0

            # Advance accepted step count
            result_status = move_res.result
            if result_status == "applied":
                step_ordinal = move_res.steps
            # else rejected moves retain previous step counter (PDF §4.1)

            game_over = move_res.game_over
            won = move_res.won
            state_after = move_res.state_after

            # J. Append Move to Official JSONL Log (Cryptographic SHA-256 Hash Chain)
            timing_dict = {
                "decide_ms": int(decide_ms),
                "api_ms": int(api_ms),
                "turn_ms": int(turn_ms),
            }
            logger.log_move(
                session_id=session_id,
                game="snake",
                step=step_ordinal if step_ordinal > 0 else 1,
                state_before=state_before,
                action=final_action,
                state_after=state_after,
                result=result_status,
                decision=decision_block,
                timing=timing_dict,
            )

            # Progress output
            score = state_after.get("score", 0)
            food_eaten = state_after.get("foodEaten", state_after.get("foods_eaten", 0))
            best_ev = ev_breakdown.get(proposed_action, {}).get("ev", 0.0)
            print(
                f"Turn {total_turns:3d} | Step {step_ordinal:3d}/{max_steps} | "
                f"Action: {final_action.upper():5s} ({source}) | Result: {result_status.upper():8s} | "
                f"Foods: {food_eaten} | Score: {score} | EV: {best_ev:+.2f} | "
                f"Decide: {decide_ms:.1f}ms | API: {api_ms:.1f}ms"
            )

            state_before = state_after

            if game_over:
                print(f"\n[!] Game Over detected from server (done=True). Won: {won}")
                break

    finally:
        logger.close_session(session_id)
        client.close()

    # Save final state JSON
    final_state_path = Path(log_dir) / f"final_state_{session_id}.json"
    with open(final_state_path, "w", encoding="utf-8") as f:
        json.dump(state_before, f, indent=2)

    final_score = state_before.get("score", 0)
    final_foods = state_before.get("foodEaten", state_before.get("foods_eaten", 0))

    print("\n" + "=" * 80)
    print("🏁 SESSION COMPLETE")
    print(f"   Session ID:     {session_id}")
    print(f"   Total Turns:    {total_turns}")
    print(f"   Accepted Steps: {step_ordinal}")
    print(f"   Foods Eaten:    {final_foods} / 15")
    print(f"   Final Score:    {final_score}")
    print(f"   Win Status:     {'🏆 WON!' if won else 'TERMINATED'}")
    print(f"   Fallback Rate:  {fallback_count}/{total_turns} ({fallback_count/max(1, total_turns)*100:.1f}%)")
    print(f"   Official Log:   {log_path.resolve()}")
    print(f"   Final State:    {final_state_path.resolve()}")
    print("=" * 80 + "\n")

    return {
        "session_id": session_id,
        "turns": total_turns,
        "steps": step_ordinal,
        "foods_eaten": final_foods,
        "score": final_score,
        "won": won,
        "log_path": str(log_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live TopGameAI Arena Snake Runner — Real Laya Model")
    parser.add_argument("--agent-name", default="ProgrammablePros", help="Agent name")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_SNAKE, help="Maximum steps (cap 300)")
    parser.add_argument("--log-dir", default="logs/snake", help="Log output directory")
    args = parser.parse_args()

    try:
        run_live_snake_session(
            agent_name=args.agent_name,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
        )
        return 0
    except Exception as exc:
        print(f"\n[!] RUNNER FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
