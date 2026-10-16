"""run_2048_live.py — Production Live Runner for 2048 on TopGameAI Arena.

HACKATHON DAY PRODUCTION IMPLEMENTATION:
1. Connects to the REAL TopGameAI Arena API: https://topgameai.com/api/arena
2. Uses the REAL Laya model: convaiinnovations/laya
3. Autonomous decision loop:
   Board Compression -> Real Laya Inference -> Multi-term EV -> Anti-No-Op Safety -> Arena Action Dispatch -> Hash-Chained JSONL Logging
4. Logs each move with SHA-256 cryptographic hash chain adhering strictly to Genesis 2.0 competition rules.
"""

from __future__ import annotations

import argparse
import json
import math
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
from game2048_brain import (
    Board2048Compressor,
    Board2048EVCalculator,
    Board2048HypothesisBuilder,
    parse_2048_state,
)
from game_clients import Game2048Client, MoveResult
from laya_agent import LayaAgent
from official_logger import OfficialLogger

LEGAL_ACTIONS: list[str] = ["up", "down", "left", "right"]
MAX_STEPS_2048: int = 2000


def run_live_2048_session(
    agent_name: str = "ProgrammablePros",
    max_steps: int = MAX_STEPS_2048,
    log_dir: str = "logs/2048",
) -> dict[str, Any]:
    """Execute a real live 2048 arena game with the real Laya neural engine."""
    print("\n" + "=" * 80)
    print("🚀 HACKATHON DAY MODE: REAL LAYA + REAL TOPGAMEAI 2048 API.")
    print("   Team: ProgrammablePros | Track: EVA 02 (Genesis 2.0) | STAGE 2")
    print("   Mode: REAL LAYA MODEL + REAL TOPGAMEAI API (PRODUCTION ONLY)")
    print("=" * 80 + "\n")

    # 1. Initialize Real Laya Agent
    # Fails loudly if model weights cannot be loaded
    laya_engine = LayaAgent(model_id="convaiinnovations/laya")

    # 2. Initialize Real TopGameAI Arena 2048 Client
    # Connects directly to https://topgameai.com/api/arena
    client = Game2048Client(agent_name=agent_name)

    # 3. Initialize Decision & Perception Pipeline
    compressor = Board2048Compressor()
    hypo_builder = Board2048HypothesisBuilder()
    ev_calc = Board2048EVCalculator()
    compliance = ComplianceLayer()

    # 4. Initialize Official SHA-256 Hash-Chained Logger
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = OfficialLogger(log_dir=log_dir)

    # 5. Start Real Arena Session
    print("\n[*] Contacting TopGameAI Arena: POST /sessions (game=2048) ...")
    try:
        session_id, state_before = client.start_game(agent_name=agent_name)
    except Exception as exc:
        raise RuntimeError(f"FATAL: Failed to create TopGameAI 2048 arena session: {exc}") from exc

    log_path = logger.start_session(session_id=session_id, game_name="2048")
    print(f"[+] Real Session Created: {session_id}")
    print(f"[+] Logging to Official JSONL: {log_path.resolve()}\n")

    # Initial State Overview
    parsed_init = parse_2048_state(state_before)
    print(f"[*] Initial 4x4 Grid:\n{parsed_init['board']}")
    print(f"[*] Initial Max Tile: {parsed_init['max_tile']} | Empty: {parsed_init['empty_count']}")
    print(f"[*] Max in Bottom-Left Corner: {bool(parsed_init['corner'])}")
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

            # B. Hypothesis Questions
            questions_dict = hypo_builder.build_hypotheses(state_before, compressed_ctx)
            question_strings = hypo_builder.build_hypothesis_strings(state_before, compressed_ctx)

            # C. Query the REAL Laya Model (ZERO SYNTHETIC DATA)
            decide_start = time.perf_counter()
            raw_laya_output = laya_engine.predict(
                compressed_state=compressed_ctx,
                questions=questions_dict,
            )

            # Reconstruct per-action progress and danger probabilities
            raw_probs_by_action: dict[str, dict[str, float]] = {}
            primary_progress_probs: dict[str, float] = {}
            for act in LEGAL_ACTIONS:
                p_prog = float(raw_laya_output["probabilities"][f"q_{act}_progress"])
                p_dang = float(raw_laya_output["probabilities"][f"q_{act}_danger"])
                raw_probs_by_action[act] = {"progress": p_prog, "danger": p_dang}
                primary_progress_probs[act] = p_prog

            # D. Multi-Term Expected Value Calculation (Includes Corner Anchoring Heuristic)
            proposed_action, ev_breakdown = ev_calc.calculate(
                state=state_before,
                raw_probs=raw_probs_by_action,
            )

            # E. Anti-No-Op Safety Interceptor: Verify that proposed action changes board
            current_board = parse_2048_state(state_before)["board"]
            final_action = proposed_action
            source = "laya"

            simulated_next = Board2048EVCalculator.simulate_slide_board(current_board, proposed_action)
            if simulated_next == current_board:
                # Proposed action would produce no board change; select best alternative that does change board
                found_valid = False
                for alt_action in ("left", "down", "up", "right"):
                    if Board2048EVCalculator.simulate_slide_board(current_board, alt_action) != current_board:
                        final_action = alt_action
                        source = "fallback"
                        fallback_count += 1
                        found_valid = True
                        compressor.record_failure(proposed_action)
                        print(f"[!] SAFETY FALLBACK: No-op '{proposed_action}' blocked -> Chose '{final_action}'")
                        break

                if not found_valid:
                    # No moves can change the board; game is over
                    print("[!] No legal moves available on current 2048 board.")
                    game_over = True
                    break
            else:
                compressor.clear_failure()

            # Record action in EV history for repeat-penalty tracking
            ev_calc.record_move(final_action)

            decide_ms = (time.perf_counter() - decide_start) * 1000.0

            # F. Normalize Primary Probabilities for the Official Log (PDF §3.2)
            normalized_probs = compliance.normalize_laya_probabilities(primary_progress_probs)

            # G. Build Canonical Decision Block
            decision_block = compliance.build_decision_block(
                questions=question_strings,
                normalized_probs=normalized_probs,
                chosen_action=final_action,
                source=source,
            )

            # H. Dispatch Move to the REAL TopGameAI Arena
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

            # I. Append Move to Official JSONL Log (Cryptographic SHA-256 Hash Chain)
            timing_dict = {
                "decide_ms": int(decide_ms),
                "api_ms": int(api_ms),
                "turn_ms": int(turn_ms),
            }
            logger.log_move(
                session_id=session_id,
                game="2048",
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
            parsed_after = parse_2048_state(state_after)
            best_tile = parsed_after["max_tile"]
            best_ev = ev_breakdown.get(proposed_action, {}).get("ev", 0.0)
            print(
                f"Turn {total_turns:3d} | Step {step_ordinal:3d}/{max_steps} | "
                f"Action: {final_action.upper():5s} ({source}) | Result: {result_status.upper():8s} | "
                f"Max Tile: {best_tile:4d} | Score: {score:5d} | EV: {best_ev:+.2f} | "
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

    parsed_final = parse_2048_state(state_before)
    final_score = state_before.get("score", 0)
    final_tile = parsed_final["max_tile"]

    # Official scoring formula for 2048:
    # Points = min(1000, max(0, (log2(best_tile) - 6) * 143 + floor(score/1000)))
    log2_tile = int(math.log2(final_tile)) if final_tile >= 2 else 0
    calculated_points = min(1000, max(0, (log2_tile - 6) * 143 + int(final_score // 1000))) if log2_tile >= 6 else 0

    print("\n" + "=" * 80)
    print("🏁 2048 SESSION COMPLETE")
    print(f"   Session ID:     {session_id}")
    print(f"   Total Turns:    {total_turns}")
    print(f"   Accepted Steps: {step_ordinal}")
    print(f"   Max Tile:       {final_tile} (2^{log2_tile})")
    print(f"   Game Score:     {final_score}")
    print(f"   Official Points:{calculated_points} / 1000")
    print(f"   Win Status:     {'🏆 WON (2048 reached)!' if won else 'TERMINATED'}")
    print(f"   Fallback Rate:  {fallback_count}/{total_turns} ({fallback_count/max(1, total_turns)*100:.1f}%)")
    print(f"   Official Log:   {log_path.resolve()}")
    print(f"   Final State:    {final_state_path.resolve()}")
    print("=" * 80 + "\n")

    return {
        "session_id": session_id,
        "turns": total_turns,
        "steps": step_ordinal,
        "max_tile": final_tile,
        "score": final_score,
        "points": calculated_points,
        "won": won,
        "log_path": str(log_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live TopGameAI Arena 2048 Runner — Real Laya Model")
    parser.add_argument("--agent-name", default="ProgrammablePros", help="Agent name")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_2048, help="Maximum steps (cap 2000)")
    parser.add_argument("--log-dir", default="logs/2048", help="Log output directory")
    args = parser.parse_args()

    try:
        run_live_2048_session(
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
