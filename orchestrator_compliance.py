"""Orchestrator Compliance Module — Genesis 2.0 EVA 02.

Master game loop integrating:
  - SnakeClient / Game2048Client (HTTP REST)
  - OfficialLogger (SHA-256 hash chain JSONL)
  - ComplianceLayer (probability normalisation + safety)
  - LayaAgent (hypothesis-driven EV pipeline)

Timing uses ``time.perf_counter()`` exclusively per PDF §6:
  - decide_ms : state received → Laya probability returned
  - api_ms    : move POST sent → game response received (from MoveResult)
  - turn_ms   : end of prev move response → end of this move response
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from advanced_ev_calculator import AdvancedEVCalculator
from compliance_layer import ComplianceLayer
from game_clients import Game2048Client, MoveResult, SnakeClient
from hypothesis_engine import HypothesisEngine
from laya_agent import LayaAgent
from official_logger import OfficialLogger
from subgoal_manager import Subgoal, SubgoalManager
from world_memory import WorldMemory

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LEGAL_ACTIONS: list[str] = ["up", "down", "left", "right"]
MAX_STEPS_SNAKE: int = 500    # Hard cap to prevent runaway sessions
MAX_STEPS_2048: int = 2000
LOG_DIR: str = "logs"


# -----------------------------------------------------------------------------
# play_compliant_game
# -----------------------------------------------------------------------------

def play_compliant_game(
    game_type: str,
    api_key: str = "",
    agent_name: str = "ProgrammablePros",
    use_mock_laya: bool = True,
    log_dir: str = LOG_DIR,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    """Run a fully compliant game session with hash-chained logging.

    Sequence per step (PDF §6):
    1. Record ``turn_start``.
    2. Get ``state_before``.
    3. Record ``decide_start`` → run EV pipeline → record ``decide_ms``.
    4. Check Snake safety fallback.
    5. Normalise probabilities → build decision block.
    6. Record ``api_start`` → POST move → record ``api_ms`` from MoveResult.
    7. Calculate ``turn_ms`` = (perf_counter() - prev_turn_end) * 1000.
    8. Call ``logger.log_move(…)`` with all required fields.
    9. If game ended → ``logger.save_final_state(…)``.

    Args:
        game_type:           "snake" or "2048" (case-insensitive).
        api_key:             Bearer token for the TopGameAI API.
        agent_name:          Agent display name.
        use_mock_laya:       If True, runs Laya in offline mock mode.
        log_dir:             Directory for JSONL log output.
        max_steps_override:  Override the default step cap.

    Returns:
        Summary dict with keys: session_id, total_steps, fallback_rate,
        log_path, final_state_path, game_over.

    Raises:
        ValueError: If game_type is not "snake" or "2048".
    """
    game_type_lower = game_type.strip().lower()
    if game_type_lower not in ("snake", "2048"):
        raise ValueError(f"Unsupported game_type: '{game_type}'. Must be 'snake' or '2048'.")

    max_steps = max_steps_override or (
        MAX_STEPS_SNAKE if game_type_lower == "snake" else MAX_STEPS_2048
    )

    # -- 1. Initialise game client ---------------------------------------------
    if game_type_lower == "snake":
        client: SnakeClient | Game2048Client = SnakeClient(
            api_key=api_key, agent_name=agent_name
        )
    else:
        client = Game2048Client(api_key=api_key, agent_name=agent_name)

    session_id, state_before = client.start_game(agent_name)
    game_name: str = game_type_lower  # "snake" or "2048" per PDF

    # -- 2. Initialise subsystems ----------------------------------------------
    logger = OfficialLogger(log_dir=log_dir)
    log_path: Path = logger.start_session(session_id, game_name)

    compliance = ComplianceLayer()
    laya = LayaAgent(use_mock=use_mock_laya)
    subgoal_mgr = SubgoalManager()
    ev_calc = AdvancedEVCalculator()
    memory = WorldMemory()
    hypo_engine = HypothesisEngine()

    logged_lines: list[dict[str, Any]] = []
    step: int = 0

    # Timing anchor: no previous move response on turn 1
    prev_turn_end: float = time.perf_counter()

    # -- 3. Main game loop -----------------------------------------------------
    # try/finally guarantees logger + client are always closed, preventing
    # Windows file-handle leaks when exceptions propagate out of the loop.
    try:
        while step < max_steps:
            step += 1
            turn_start: float = time.perf_counter()  # noqa: F841

            # -- 3a. Subgoal arbitration ---------------------------------------
            decide_start: float = time.perf_counter()
            try:
                subgoal: Subgoal = subgoal_mgr.select_subgoal(state_before, memory)
            except Exception:
                subgoal = Subgoal(
                    target_type="EXPLORE",
                    coordinates=None,
                    description="Fallback exploration subgoal.",
                )

            # -- 3b. State compression (scalar normalisation) ------------------
            # laya.compress_state() uses arithmetic on player_pos and expects a
            # scalar int. API states may return player_pos as a [x, y] list;
            # we collapse it to a single int before passing it to laya.
            _raw_pos = state_before.get("player_pos", 0)
            if isinstance(_raw_pos, (list, tuple)) and len(_raw_pos) >= 2:
                _scalar_state: dict[str, Any] = dict(state_before)
                _scalar_state["player_pos"] = int(_raw_pos[0]) + int(_raw_pos[1])
            else:
                _scalar_state = state_before

            compressed: str = laya.compress_state(_scalar_state)

            # -- 3c. Hypothesis generation -------------------------------------
            hypotheses = hypo_engine.generate_hypotheses(
                compressed_context=compressed,
                legal_actions=LEGAL_ACTIONS,
                subgoal=subgoal,
                memory=memory,
            )
            flat_queries = hypo_engine.flatten_queries(hypotheses)
            question_strings: list[str] = [q for (_, _, q) in flat_queries]

            # -- 3d. Laya probability inference --------------------------------
            # compressed_state_override bypasses laya's internal compress_state()
            # so it never tries to do arithmetic on the raw API state.
            raw_probs: dict[str, float] = laya.decide(
                state=_scalar_state,
                possible_actions=LEGAL_ACTIONS,
                compressed_state_override=compressed,
            )

            # -- 3e. EV calculation --------------------------------------------
            player_pos: Any = state_before.get("player_pos", (0, 0))
            chosen_action, ev_breakdown = ev_calc.calculate(
                current_pos=player_pos,
                legal_actions=LEGAL_ACTIONS,
                probs={
                    d: {
                        "is_legal": raw_probs.get(d, 0.5),
                        "toward_subgoal": raw_probs.get(d, 0.5),
                        "is_dangerous": max(0.0, 0.5 - raw_probs.get(d, 0.5)),
                        "is_novel": 0.1,
                    }
                    for d in LEGAL_ACTIONS
                },
                recent_failures=laya.recent_outcomes,
            )

            decide_ms: float = (time.perf_counter() - decide_start) * 1000.0

            # -- 3f. Snake safety fallback -------------------------------------
            source: str = "laya"
            if game_type_lower == "snake":
                safe_alternative = compliance.check_snake_safety_fallback(
                    current_state=state_before,
                    proposed_action=chosen_action,
                )
                if safe_alternative is not None:
                    chosen_action = safe_alternative
                    source = "fallback"

            # -- 3g. Normalise probabilities -----------------------------------
            ev_scores_for_norm = {d: ev_breakdown.get(d, {}) for d in LEGAL_ACTIONS}
            normalized_probs: dict[str, float] = compliance.normalize_probabilities_for_log(
                ev_scores_for_norm
            )

            # -- 3h. Build decision block --------------------------------------
            decision_block: dict[str, Any] = compliance.build_decision_block(
                questions=question_strings,
                normalized_probs=normalized_probs,
                chosen_action=chosen_action,
                source=source,
            )

            # -- 3i. Send move (api_ms timed inside MoveResult) ----------------
            move_result: MoveResult = client.send_move(
                session_id=session_id,
                action=chosen_action,
                **({"state_before": state_before} if game_type_lower == "2048" else {}),
            )
            state_after = move_result.state_after
            result = move_result.result

            # -- 3j. Compute turn_ms -------------------------------------------
            turn_end: float = time.perf_counter()
            turn_ms: float = (turn_end - prev_turn_end) * 1000.0
            prev_turn_end = turn_end

            # -- 3k. Log the move ----------------------------------------------
            timing: dict[str, float] = {
                "decide_ms": round(decide_ms, 3),
                "api_ms": round(move_result.api_ms, 3),
                "turn_ms": round(turn_ms, 3),
            }
            log_line = logger.log_move(
                session_id=session_id,
                game=game_name,
                step=step,
                state_before=state_before,
                action=chosen_action,
                state_after=state_after,
                result=result,
                decision=decision_block,
                timing=timing,
            )
            logged_lines.append(log_line)

            # -- 3l. Update Laya short-term memory -----------------------------
            laya.record_outcome(
                action=chosen_action,
                prev_state=_scalar_state,
                new_state=state_after,
                success=(result == "applied"),
                details=f"step={step} result={result}",
            )

            # -- 3m. Advance state ---------------------------------------------
            state_before = state_after

            # -- 3n. Game-over handling ----------------------------------------
            if move_result.game_over:
                final_state = client.get_final_state(session_id)
                final_state_path = logger.save_final_state(session_id, final_state)
                logger.close_session(session_id)
                client.close()
                fallback_rate = compliance.compute_fallback_rate(logged_lines)
                return {
                    "session_id": session_id,
                    "total_steps": step,
                    "fallback_rate": round(fallback_rate, 4),
                    "log_path": str(log_path),
                    "final_state_path": str(final_state_path),
                    "game_over": True,
                }

        # -- 4. Step-cap reached -----------------------------------------------
        final_state_cap = client.get_final_state(session_id)
        final_path_cap = logger.save_final_state(session_id, final_state_cap)
        logger.close_session(session_id)
        client.close()
        fallback_rate = compliance.compute_fallback_rate(logged_lines)
        return {
            "session_id": session_id,
            "total_steps": step,
            "fallback_rate": round(fallback_rate, 4),
            "log_path": str(log_path),
            "final_state_path": str(final_path_cap),
            "game_over": False,
            "note": "Terminated at step cap.",
        }

    finally:
        # Idempotent — safe to call twice if game ended cleanly above.
        # Ensures file handles are released on Windows even after exceptions.
        logger.close_session(session_id)
        client.close()


# -----------------------------------------------------------------------------
# __main__: Simulate 3 Snake turns with mock state, verify hash chain
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    print("=" * 60)
    print("Genesis 2.0 Compliance Simulation — 3-Turn Snake Test")
    print("=" * 60)

    MOCK_SESSION_ID = f"SIM-{uuid.uuid4().hex[:8].upper()}"

    def _make_snake_state(step_n: int, direction: str = "up") -> dict[str, Any]:
        return {
            "board": [[0] * 10 for _ in range(10)],
            "direction": direction,
            "score": step_n * 10,
            "step": step_n,
            "game_over": step_n >= 3,  # Trigger game-over after step 3
            "player_pos": [5 + step_n, 5],
        }

    # Server responses: 1 start + 3 moves + 1 final GET
    _mock_responses: list[dict[str, Any]] = [
        {"session_id": MOCK_SESSION_ID, "state": _make_snake_state(0)},
        {"result": "applied", "state": _make_snake_state(1)},
        {"result": "applied", "state": _make_snake_state(2)},
        {"result": "applied", "state": _make_snake_state(3)},
        {"state": _make_snake_state(3, "up")},
    ]
    _mock_idx = 0

    def _mock_post(*args: Any, **kwargs: Any) -> MagicMock:
        global _mock_idx
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = lambda: None
        r.json.return_value = _mock_responses[_mock_idx]
        _mock_idx += 1
        return r

    def _mock_get(*args: Any, **kwargs: Any) -> MagicMock:
        global _mock_idx
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = lambda: None
        r.json.return_value = _mock_responses[_mock_idx]
        _mock_idx += 1
        return r

    # Use a named tempdir that we manage manually to avoid Windows lock issues
    tmpdir = tempfile.mkdtemp(prefix="genesis2_test_")
    try:
        with (
            patch("requests.Session.post", side_effect=_mock_post),
            patch("requests.Session.get", side_effect=_mock_get),
        ):
            summary = play_compliant_game(
                game_type="snake",
                api_key="mock-key",
                log_dir=tmpdir,
                use_mock_laya=True,
            )

        print(f"\nSession ID   : {summary['session_id']}")
        print(f"Total Steps  : {summary['total_steps']}")
        print(f"Fallback Rate: {summary['fallback_rate'] * 100:.1f}%")
        print(f"Log File     : {summary['log_path']}")
        print(f"Final State  : {summary['final_state_path']}")

        # -- Verify hash chain programmatically --------------------------------
        print("\n-- Hash Chain Verification --------------------------")
        log_file = Path(summary["log_path"])
        valid, msg = OfficialLogger.verify_log_file(log_file)
        if valid:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            raise SystemExit(1)

        # Print each log line and assert PDF invariants
        print("\n-- Log Line Summary -")
        with open(log_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                entry = json.loads(raw_line.strip())
                prob_sum = sum(entry["decision"]["probabilities"].values())
                print(
                    f"  Step {entry['step']:>3} | action={entry['action']:<6}"
                    f" | result={entry['result']:<9}"
                    f" | source={entry['decision']['source']:<9}"
                    f" | prob_sum={prob_sum:.6f}"
                    f" | confidence={entry['decision']['confidence']:.4f}"
                    f" | hash={entry['hash'][:12]}..."
                )
                # Assert every PDF invariant
                assert entry["result"] in ("applied", "rejected"), \
                    f"Invalid result: {entry['result']}"
                assert 0.98 <= prob_sum <= 1.02, \
                    f"Prob sum out of range: {prob_sum}"
                assert entry["decision"]["chosen"] == entry["action"], \
                    f"chosen ({entry['decision']['chosen']}) != action ({entry['action']})"
                assert entry["decision"]["confidence"] == \
                    entry["decision"]["probabilities"][entry["action"]], \
                    "confidence != probabilities[action]"
                assert entry["decision"]["source"] in ("laya", "fallback"), \
                    f"Invalid source: {entry['decision']['source']}"
                assert "decide_ms" in entry["timing"], "Missing decide_ms"
                assert "api_ms" in entry["timing"], "Missing api_ms"
                assert "turn_ms" in entry["timing"], "Missing turn_ms"

        print(f"\n[orchestrator_compliance] ALL INVARIANTS PASSED")
        print(f"  Fallback budget: {summary['fallback_rate'] * 100:.1f}% (<=10% required)")

    finally:
        # Clean up temp directory (files should be closed by this point)
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass  # Non-fatal on Windows if handles are still open
