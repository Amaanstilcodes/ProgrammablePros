"""Unified Runner for Local Snake & 2048 Games — Genesis 2.0 Track EVA 02.

Executes local Snake and 2048 sessions using the full decision pipeline:
State Compression -> Hypothesis Formulation -> Laya Model Scoring -> EV Calculation -> PDF Compliance -> SHA-256 Hash Chain.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any
from typing import Any

from compliance_layer import ComplianceLayer
from core.laya_backend import LayaBackend, LocalLayaBackend
from game2048_brain import (
    Board2048Compressor,
    Board2048EVCalculator,
    Board2048HypothesisBuilder,
)
from local_game_clients import LocalGame2048Client, LocalSnakeClient
from official_logger import OfficialLogger
from snake_brain import (
    SnakeCompressor,
    SnakeEVCalculator,
    SnakeHypothesisBuilder,
)


def score_hypotheses_with_laya(
    laya_model: LayaBackend,
    context: str,
    questions: Any,
) -> dict[str, dict[str, float]]:
    """Score hypothesis questions using the Laya model backend.

    Handles both a flat list[str] and the Dict[str, Dict] format
    returned by SnakeHypothesisBuilder / Board2048HypothesisBuilder.
    """
    ACTIONS = ("up", "down", "left", "right")

    # Normalise to a flat list of question strings
    if isinstance(questions, dict):
        # Format: {'q_up': {'instructions': '...'}, 'q_down': {...}, ...}
        # or      {'up': {'instructions': '...'}, ...}
        flat_questions: list[str] = []
        action_order: list[str] = []
        for action in ACTIONS:
            key = f"q_{action}" if f"q_{action}" in questions else action
            if key in questions:
                val = questions[key]
                text = val.get("instructions", val.get("text", str(val))) if isinstance(val, dict) else str(val)
                flat_questions.append(text)
                action_order.append(action)
        scores = laya_model.batch_predict(context, flat_questions)
        raw_probs: dict[str, dict[str, float]] = {}
        for i, action in enumerate(action_order):
            score = scores[i] if i < len(scores) else 0.5
            raw_probs[action] = {"q1": float(score), "q2": float(1.0 - score)}
        # Fill any missing actions
        for action in ACTIONS:
            if action not in raw_probs:
                raw_probs[action] = {"q1": 0.25, "q2": 0.75}
    else:
        # Legacy flat list of 8 strings: [up_q1, up_q2, down_q1, down_q2, ...]
        scores = laya_model.batch_predict(context, list(questions))
        raw_probs = {}
        for i, action in enumerate(ACTIONS):
            q1_score = scores[i * 2] if (i * 2) < len(scores) else 0.5
            q2_score = scores[i * 2 + 1] if (i * 2 + 1) < len(scores) else 0.5
            raw_probs[action] = {"q1": float(q1_score), "q2": float(q2_score)}

    return raw_probs


def run_session(game: str, seed: int | None = None) -> tuple[str, bool, dict[str, Any]]:
    """Run a single game session to completion using the real Laya backend in project."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = OfficialLogger(log_dir=str(log_dir))
    compliance = ComplianceLayer()

    # Instantiate Real Laya backend — ZERO MOCKS ALLOWED
    laya_model: LayaBackend = LocalLayaBackend()
    backend_name = "Real Laya Model (LocalLayaBackend: convaiinnovations/laya)"

    # Initialize game client and brains
    if game == "snake":
        client = LocalSnakeClient(width=20, height=20, seed=seed)
        compressor = SnakeCompressor()
        class _SnakeSubgoal:
            def get_current_subgoal(self, state):
                foods = state.get('foods_eaten', 0)
                return f"Eat food #{foods + 1} / 15 without dying"
        subgoal_mgr = _SnakeSubgoal()
        hypo_builder = SnakeHypothesisBuilder()
        ev_calc = SnakeEVCalculator()
        step_cap = 300
    elif game == "2048":
        client = LocalGame2048Client(seed=seed)
        compressor = Board2048Compressor()
        class _2048Subgoal:
            def get_current_subgoal(self, state):
                best = state.get('best_tile', 0)
                return f"Merge tiles to reach {best * 2 if best else 512}"
        subgoal_mgr = _2048Subgoal()
        hypo_builder = Board2048HypothesisBuilder()
        ev_calc = Board2048EVCalculator()
        step_cap = 2000
    else:
        raise ValueError(f"Unknown game: {game}")

    session_id, state = client.start_game()
    log_path = logger.start_session(session_id=session_id, game_name=game)

    print(f"\n========================================================")
    print(f"  Starting Local Session: {session_id} | Game: {game.upper()}")
    print(f"  Backend: {backend_name}")
    print(f"  Log File: {log_path}")
    print(f"========================================================\n")

    step_ordinal = 0
    fallback_count = 0
    total_moves = 0

    while not state.get("game_over", False) and step_ordinal < step_cap:
        t_start = time.perf_counter()

        # a. Compress state into dense context string
        compressed = compressor.compress(state)

        # b. Get current subgoal
        subgoal = subgoal_mgr.get_current_subgoal(state)

        # c. Build 8 typed hypotheses
        questions = hypo_builder.build_hypotheses(state, compressed)

        # d. Query Laya Model (NO hardcoded values)
        t_laya0 = time.perf_counter()
        raw_probs = score_hypotheses_with_laya(laya_model, compressed, questions)
        laya_ms = (time.perf_counter() - t_laya0) * 1000.0

        # e. Calculate EV using Laya probabilities
        proposed_action, ev_breakdown = ev_calc.calculate(state, raw_probs, subgoal)

        # f. Check safety fallback
        final_action = proposed_action
        source = "laya"
        if game == "snake":
            fallback = compliance.check_snake_safety_fallback(state, proposed_action)
            if fallback is not None:
                final_action = fallback
                source = "fallback"
                fallback_count += 1

        t_decide = (time.perf_counter() - t_start) * 1000.0

        # g. Normalize PRIMARY Laya probabilities for the log (PDF §3.2)
        # Primary hypothesis is Q1 (progress towards subgoal)
        primary_laya_probs = {a: float(raw_probs[a]["q1"]) for a in ("up", "down", "left", "right")}
        normalized_probs = compliance.normalize_laya_probabilities(primary_laya_probs)

        # h. Build official decision block (confidence == normalized_probs[chosen])
        decision_block = compliance.build_decision_block(
            questions=questions,
            normalized_probs=normalized_probs,
            chosen_action=final_action,
            source=source,
        )

        # i. Send move to game client
        state_before = dict(state)
        move_resp = client.send_move(session_id, final_action, state_before=state_before)
        state_after = move_resp["state_after"]
        result = move_resp["result"]
        api_ms = move_resp.get("api_ms", 0.0)

        # Advance step counter if move applied
        if result == "applied":
            step_ordinal += 1

        # j. Log to OfficialLogger (cryptographic hash chain)
        total_moves += 1
        turn_ms = (time.perf_counter() - t_start) * 1000.0
        timing_dict = {
            "decide_ms": round(t_decide, 3),
            "api_ms": round(api_ms, 3),
            "turn_ms": round(turn_ms, 3),
        }

        logger.log_move(
            session_id=session_id,
            game=game,
            step=step_ordinal if step_ordinal > 0 else 1,
            state_before=state_before,
            action=final_action,
            state_after=state_after,
            result=result,
            decision=decision_block,
            timing=timing_dict,
        )

        # k. Update adaptive memory & counters
        ev_calc.record_move(final_action)
        if result == "rejected":
            compressor.record_failure(final_action)
        else:
            compressor.clear_failure()

        # l. Print turn telemetry
        p_str = " ".join(f"{a}:{int(normalized_probs[a]*100)}%" for a in ("up", "down", "left", "right"))
        ev_str = " ".join(f"{a}:{ev_breakdown[a]['ev']:+.1f}" for a in ("up", "down", "left", "right"))
        score_val = state_after.get("score", 0)
        extra_kpi = f"Foods: {state_after.get('foods_eaten', 0)}" if game == "snake" else f"Tile: {state_after.get('best_tile', 0)}"

        print(
            f"[Step {step_ordinal:02d}] Score: {score_val:3d} | {extra_kpi} | Probs: [{p_str}] "
            f"| EVs: [{ev_str}] | -> Move: '{final_action.upper()}' "
            f"(EV: {ev_breakdown[final_action]['ev']:+.2f}) | Source: {source} (Laya: {laya_ms:.1f}ms)"
        )

        state = state_after

    # Finalize session
    client.close()
    logger.save_final_state(session_id, state)
    logger.close_session(session_id)

    # Verify cryptographic hash chain
    is_valid, verify_msg = OfficialLogger.verify_log_file(log_path)
    fallback_pct = (fallback_count / max(1, total_moves)) * 100.0

    print(f"\n--------------------------------------------------------")
    print(f"Session Finished: {session_id}")
    print(f"Final Score: {state.get('score', 0)}")
    if game == "snake":
        print(f"Foods Eaten: {state.get('foods_eaten', 0)} / 15")
    else:
        print(f"Best Tile: {state.get('best_tile', 0)} | Points: {state.get('points', 0)}")
    print(f"Steps: {step_ordinal} | Total Moves: {total_moves}")
    print(f"Fallback Rate: {fallback_pct:.2f}% (Limit <= 10%)")
    print(f"Hash Chain Verification: {verify_msg}")
    print(f"--------------------------------------------------------\n")

    summary = {
        "session_id": session_id,
        "game": game,
        "score": state.get("score", 0),
        "steps": step_ordinal,
        "total_moves": total_moves,
        "fallback_pct": fallback_pct,
        "is_valid": is_valid,
        "log_path": str(log_path),
    }
    return session_id, is_valid, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Snake or 2048 game session with Laya EV pipeline.")
    parser.add_argument("--game", choices=["snake", "2048"], default="snake", help="Game to run (snake or 2048)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    session_id, is_valid, summary = run_session(game=args.game, seed=args.seed)
    if not is_valid:
        print(f"FAILED: Hash chain verification failed for {session_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
