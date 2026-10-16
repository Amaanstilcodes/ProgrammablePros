"""Unified Compliance Orchestrator — Genesis 2.0 Track EVA 02.

Master competition runner coordinating:
- Modular GameBundles for Snake (Stage 1) and 2048 (Stage 2)
- MonotonicTurnTimer (time.perf_counter() precision)
- StepLedger (Configurable rejected step policy)
- Dual logging: Official hash-chain JSONL + Rich internal trace
- Rule R1-R10 enforcement (Stage gate, Laya-only decisions, <= 10% fallback)
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.advanced_ev_calculator import AdvancedEVCalculator
from core.compliance_layer import ComplianceLayer
from core.config import AppConfig
from core.laya_backend import HttpLayaBackend, LayaBackend, LocalLayaBackend, MockLayaBackend
from core.official_logger import OfficialLogger
from core.rich_trace import RichTraceLogger
from core.stage_gate import StageGate, StageLockedError
from core.step_ledger import StepLedger
from core.timing import MonotonicTurnTimer
from games.registry import get_game_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


def run_orchestrated_game(
    game: str = "snake",
    api_key: str = "",
    agent_name: str = "ProgrammablePros",
    is_mock: bool = False,
    max_steps_override: int | None = None,
    allow_stage_override: bool = False,
    rejected_step_policy: str = "repeat_last_applied",
    custom_log_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a fully compliant game session.

    Returns summary dictionary with session_id, accepted_steps, log_path, and outcome.
    """
    game_clean = game.strip().lower()

    # 1. Enforce Rule R8 (Stage Gate for 2048)
    stage_gate = StageGate()
    stage_gate.assert_stage_unlocked(game_clean, allow_override=(is_mock or allow_stage_override))

    # 2. Acquire Game Component Suite
    bundle = get_game_bundle(
        game=game_clean,
        api_key=api_key,
        agent_name=agent_name,
        is_mock=is_mock,
    )
    max_steps = max_steps_override or bundle.max_steps

    # 3. Initialize Engine Components
    step_ledger = StepLedger(
        policy=rejected_step_policy,  # type: ignore[arg-type]
        max_accepted_steps=max_steps,
    )
    compliance = ComplianceLayer()
    ev_calc = AdvancedEVCalculator()
    laya_backend: LayaBackend = LocalLayaBackend()

    # 4. Start Game Session
    session_id, raw_state_before = bundle.client.start_game(agent_name)
    initial_response_time = time.perf_counter()
    timer = MonotonicTurnTimer(initial_turn_start=initial_response_time)

    # 5. Initialize Dual Loggers
    official_log = OfficialLogger(is_mock=is_mock)
    log_path = official_log.start_session(
        session_id=session_id,
        game=game_clean,
        agent_name=agent_name,
        custom_log_path=custom_log_path,
    )
    session_dir = official_log.session_dir or Path(f"logs/{game_clean}/{session_id}")
    rich_trace = RichTraceLogger(session_dir=session_dir, session_id=session_id)
    rich_trace.open()

    print("================================================================================")
    print(f"[*] TOPGAMEAI ORCHESTRATOR — Genesis 2.0 Track EVA 02")
    print(f"    Game: {game_clean.upper()} | Mock: {is_mock} | Agent: {agent_name}")
    print(f"    Session ID: {session_id}")
    print(f"    Official Log: {log_path}")
    print("================================================================================")

    state_before = raw_state_before
    game_over = False
    turn_counter = 0

    try:
        while not game_over and not step_ledger.is_cap_reached:
            turn_counter += 1
            timer.start_decision()

            # Parse & extract features
            parsed_state = bundle.parser.parse(state_before)
            features = bundle.features.extract_features(parsed_state, bundle.memory)
            subgoal = bundle.subgoals.arbitrate(parsed_state, bundle.memory)
            hypos = bundle.hypotheses.build_hypotheses(features, subgoal, bundle.legal_actions)

            # Query Laya backend
            # Primary hypothesis probabilities (for official log probabilities block)
            primary_probabilities: dict[str, float] = {}
            for act in bundle.legal_actions:
                q = hypos["queries_by_action"][act]["toward_subgoal"]
                primary_probabilities[act] = laya_backend.predict_probability(hypos["context"], q)

            # Full 4-way typed probabilities for EV calculation
            typed_probabilities: dict[str, dict[str, float]] = {}
            for act in bundle.legal_actions:
                typed_probabilities[act] = {
                    dim: laya_backend.predict_probability(hypos["context"], q_text)
                    for dim, q_text in hypos["queries_by_action"][act].items()
                }

            # Multi-objective Expected Value synthesis
            player_pos = features.get("player_pos", (0, 0))
            best_action, ev_breakdown = ev_calc.calculate(
                current_pos=player_pos,
                legal_actions=bundle.legal_actions,
                probs=typed_probabilities,
            )
            ev_scores = {a: ev_breakdown[a]["ev"] for a in bundle.legal_actions}

            # Action selection & safety fallback check
            chosen_action, source = bundle.policy.select_action(
                action_probabilities=primary_probabilities,
                ev_scores=ev_scores,
                legal_actions=bundle.legal_actions,
                state_before=state_before,
            )

            timer.end_decision()

            # Normalize primary action probabilities (strictly [0.98, 1.02] summing to 1.000000)
            normalized_probs = compliance.normalize_probabilities(primary_probabilities)

            # Build Canonical Decision Block (Invariant I1-I6)
            decision_block = compliance.build_decision_block(
                questions=hypos["primary_questions"],
                normalized_probs=normalized_probs,
                chosen_action=chosen_action,
                source=source,
            )

            # Dispatch Move
            timer.start_api()
            move_result = bundle.client.send_move(
                session_id=session_id,
                action=chosen_action,
                state_before=state_before,
            )
            timer.end_api(move_result.api_ms)
            timing = timer.complete_turn()

            # Record in StepLedger
            step_to_log = step_ledger.record_move(move_result.result)

            # Append Official Log Line
            log_line = official_log.log_move(
                step=step_to_log,
                state_before=state_before,
                action=chosen_action,
                state_after=move_result.state_after,
                result=move_result.result,
                decision=decision_block,
                timing=timing.to_dict(),
            )

            # Append Rich Trace Line
            rich_trace.log_trace(
                step=step_to_log,
                action=chosen_action,
                result=move_result.result,
                raw_laya_output=primary_probabilities,
                typed_probabilities=typed_probabilities,
                anchor_probes=[],
                ev_breakdown=ev_breakdown,
                subgoal=subgoal,
                features=features,
            )

            # Memory observation
            parsed_after = bundle.parser.parse(move_result.state_after)
            bundle.memory.observe(parsed_state, chosen_action, parsed_after)

            # Console Telemetry
            badge = "[LAYA]" if source == "laya" else "[FALLBACK]"
            print(
                f"[TURN {turn_counter:03d} | STEP {step_to_log:03d}] "
                f"Act: {chosen_action.upper():<5} {badge:<10} | "
                f"Res: {move_result.result.upper():<8} | Conf: {decision_block['confidence']:.4f} | "
                f"Turn: {timing.turn_ms:3d}ms [D:{timing.decide_ms:2d}ms A:{timing.api_ms:2d}ms] | "
                f"Score: {parsed_after.get('score', 0)}"
            )

            state_before = move_result.state_after
            game_over = move_result.game_over

    finally:
        # Atomic final state & manifest persistence
        final_state_path = official_log.save_final_state(state_before)
        manifest_path = official_log.write_manifest(
            agent_name=agent_name,
            outcome="completed" if game_over else ("cap_reached" if step_ledger.is_cap_reached else "aborted"),
        )
        official_log.close()
        rich_trace.close()
        bundle.client.close()

    # Post-run hash verification
    is_valid, msg = OfficialLogger.verify_log_file(log_path)
    print(f"\n[SUMMARY] Session finished. Accepted Steps: {step_ledger.accepted_count}, Total Moves: {step_ledger.total_attempts}")
    print(f"          Log Verification: {msg}")
    print(f"          Final State: {final_state_path.name}")
    print(f"          Manifest: {manifest_path.name}")

    return {
        "session_id": session_id,
        "accepted_steps": step_ledger.accepted_count,
        "total_moves": step_ledger.total_attempts,
        "game_over": game_over,
        "log_path": str(log_path),
        "final_state_path": str(final_state_path),
        "is_valid": is_valid,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Compliance Orchestrator — Genesis 2.0 Track EVA 02")
    parser.add_argument("--game", choices=["snake", "2048"], default="snake", help="Game to play (default: snake)")
    parser.add_argument("--mock", action=argparse.BooleanOptionalAction, default=True, help="Run mock offline session")
    parser.add_argument("--api-key", type=str, default=os.getenv("TOPGAME_API_KEY", ""), help="TopGameAI API key")
    parser.add_argument("--agent-name", type=str, default="ProgrammablePros", help="Agent display name")
    parser.add_argument("--max-steps", type=int, default=None, help="Override step cap")
    parser.add_argument("--allow-stage-override", action="store_true", help="Force run Stage 2 without unlock marker")
    parser.add_argument("--policy", choices=["repeat_last_applied", "pending_ordinal"], default="repeat_last_applied", help="Step ledger policy")

    args = parser.parse_args()
    run_orchestrated_game(
        game=args.game,
        api_key=args.api_key,
        agent_name=args.agent_name,
        is_mock=args.mock,
        max_steps_override=args.max_steps,
        allow_stage_override=args.allow_stage_override,
        rejected_step_policy=args.policy,
    )
