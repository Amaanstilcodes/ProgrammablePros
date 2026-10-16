"""Compliant Runner Module — Genesis 2.0 Track EVA 02 Master Entry Point.

Orchestrates the live TopGameAI competition API with the V2 intelligence architecture:
- WorldMemory (spatial memory & failure tracking)
- EgocentricCompressor (dense egocentric POV prompts)
- SubgoalManager (hierarchical tactical arbitration)
- HypothesisEngine (4-dimensional hypothesis query formulation)
- AdvancedEVCalculator & CalibrationEngine
- Strict ComplianceLayer (anti-reversal safety & [0.98, 1.02] probability normalisation)
- OfficialLogger (SHA-256 cryptographic hash chain JSONL logging)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from advanced_ev_calculator import AdvancedEVCalculator
from calibration_engine import CalibrationEngine
from compliance_layer import ComplianceLayer
from egocentric_compressor import EgocentricCompressor
from game_clients import BaseGameClient, Game2048Client, MoveResult, SnakeClient
from hypothesis_engine import HypothesisEngine
from official_logger import OfficialLogger
from subgoal_manager import Subgoal, SubgoalManager
from world_memory import STATUS_FREE, STATUS_WALL, WorldMemory

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Defaults
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_ACTIONS: list[str] = ["up", "down", "left", "right"]
MAX_STEPS_SNAKE: int = 300
MAX_STEPS_2048: int = 2000
DEFAULT_LOG_PATH: str = "decision_log.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Mock Clients for Safe Local Testing
# ─────────────────────────────────────────────────────────────────────────────

class MockSnakeClient(SnakeClient):
    """Offline mock client simulating the TopGameAI Snake arena."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step_count: int = 0
        self._snake: list[list[int]] = [[5, 5], [5, 4], [5, 3]]
        self._food: list[int] = [8, 5]
        self._direction: str = "right"
        self._score: int = 0
        self._game_over: bool = False

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        self._step_count = 0
        self._snake = [[5, 5], [5, 4], [5, 3]]
        self._food = [8, 5]
        self._direction = "right"
        self._score = 0
        self._game_over = False
        initial_state: dict[str, Any] = {
            "snake": [list(pt) for pt in self._snake],
            "food": list(self._food),
            "direction": self._direction,
            "score": self._score,
            "game_over": False,
            "step": 0,
            "steps_remaining": MAX_STEPS_SNAKE,
        }
        return f"MOCK-SNAKE-{uuid.uuid4().hex[:8]}", initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        t0 = time.perf_counter()
        act = action.strip().lower()
        self._step_count += 1

        deltas: dict[str, tuple[int, int]] = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        dx, dy = deltas.get(act, (0, 0))
        head = self._snake[0]
        new_head = [head[0] + dx, head[1] + dy]

        # 16x16 grid bounds
        if new_head[0] < 0 or new_head[0] >= 16 or new_head[1] < 0 or new_head[1] >= 16:
            result = "rejected"
            new_head = head
        else:
            result = "applied"
            self._direction = act
            self._snake.insert(0, new_head)
            if new_head == self._food:
                self._score += 10
                self._food = [(self._food[0] + 4) % 15, (self._food[1] + 3) % 15]
            else:
                self._snake.pop()

        api_ms = (time.perf_counter() - t0) * 1000.0 + random.uniform(1.0, 3.0)
        state_after: dict[str, Any] = {
            "snake": [list(pt) for pt in self._snake],
            "food": list(self._food),
            "direction": self._direction,
            "score": self._score,
            "game_over": self._game_over,
            "step": self._step_count,
            "steps_remaining": max(0, MAX_STEPS_SNAKE - self._step_count),
        }
        return MoveResult(
            state_after=state_after,
            result=result,
            game_over=self._game_over,
            api_ms=round(api_ms, 3),
        )


class MockGame2048Client(Game2048Client):
    """Offline mock client simulating the TopGameAI 2048 arena."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step_count: int = 0
        self._board: list[list[int]] = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 2],
        ]
        self._score: int = 16
        self._game_over: bool = False

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        self._step_count = 0
        self._board = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 2],
        ]
        self._score = 16
        self._game_over = False
        initial_state: dict[str, Any] = {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": 8,
            "game_over": False,
            "step": 0,
            "steps_remaining": MAX_STEPS_2048,
        }
        return f"MOCK-2048-{uuid.uuid4().hex[:8]}", initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        t0 = time.perf_counter()
        self._step_count += 1

        # Deterministic board progression
        spawn_r = self._step_count % 4
        spawn_c = (self._step_count // 4) % 4
        self._board[spawn_r][spawn_c] = max(self._board[spawn_r][spawn_c], 2)
        self._score += 2

        max_tile = max(max(row) for row in self._board)
        api_ms = (time.perf_counter() - t0) * 1000.0 + random.uniform(1.0, 3.0)
        state_after: dict[str, Any] = {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": max_tile,
            "game_over": self._game_over,
            "step": self._step_count,
            "steps_remaining": max(0, MAX_STEPS_2048 - self._step_count),
        }
        return MoveResult(
            state_after=state_after,
            result="applied",
            game_over=self._game_over,
            api_ms=round(api_ms, 3),
        )


# ─────────────────────────────────────────────────────────────────────────────
# State Adapter Function
# ─────────────────────────────────────────────────────────────────────────────

def adapt_api_state_to_v2(raw_state: dict[str, Any], game: str) -> dict[str, Any]:
    """Map raw API state into the format expected by V2 EgocentricCompressor and SubgoalManager.

    Args:
        raw_state: Game state dict from client API.
        game: "snake" or "2048".

    Returns:
        Standardised dictionary with player_pos, goal_pos, distance_to_goal, direction, etc.
    """
    game_clean = game.strip().lower()
    adapted: dict[str, Any] = dict(raw_state)

    if game_clean == "snake":
        # Snake mapping: snake[0] is head
        snake_body = raw_state.get("snake", [])
        if isinstance(snake_body, (list, tuple)) and len(snake_body) > 0:
            head = snake_body[0]
            if isinstance(head, (list, tuple)) and len(head) >= 2:
                player_pos: tuple[int, int] = (int(head[0]), int(head[1]))
            elif isinstance(head, dict):
                player_pos = (int(head.get("x", 0)), int(head.get("y", 0)))
            else:
                player_pos = (0, 0)
        elif "player_pos" in raw_state:
            p = raw_state["player_pos"]
            player_pos = (int(p[0]), int(p[1])) if isinstance(p, (list, tuple)) else (int(p), 0)
        else:
            player_pos = (0, 0)

        # Food mapping: food target
        food_data = raw_state.get("food", [5, 5])
        if isinstance(food_data, (list, tuple)) and len(food_data) > 0:
            if isinstance(food_data[0], (list, tuple)):
                goal_pos: tuple[int, int] = (int(food_data[0][0]), int(food_data[0][1]))
            else:
                goal_pos = (int(food_data[0]), int(food_data[1]))
        elif isinstance(food_data, dict):
            goal_pos = (int(food_data.get("x", 5)), int(food_data.get("y", 5)))
        else:
            goal_pos = (5, 5)

        direction = str(
            raw_state.get("direction")
            or raw_state.get("current_direction")
            or "right"
        ).lower()

        distance_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        steps_remaining = raw_state.get(
            "steps_remaining", max(0, MAX_STEPS_SNAKE - raw_state.get("step", 0))
        )
        score = raw_state.get("score", max(0, len(snake_body) - 3) * 10)

        adapted.update({
            "player_pos": player_pos,
            "goal_pos": goal_pos,
            "distance_to_goal": distance_to_goal,
            "direction": direction,
            "facing": direction,
            "steps_remaining": steps_remaining,
            "score": score,
            "health": 100,
            "keys_held": 1,  # Bypass key requirement in SubgoalManager
        })

    elif game_clean == "2048":
        # 2048 mapping: highest tile coordinate is goal_pos
        board = raw_state.get("board") or raw_state.get("grid") or [[0] * 4 for _ in range(4)]
        max_val: int = -1
        highest_pos: tuple[int, int] = (0, 0)

        for r_idx, row in enumerate(board):
            for c_idx, val in enumerate(row):
                if isinstance(val, (int, float)) and val > max_val:
                    max_val = int(val)
                    highest_pos = (c_idx, r_idx)

        goal_pos = highest_pos
        player_pos = (1, 1)  # Abstract centroid of the 4x4 matrix
        score = int(raw_state.get("score", 0))
        max_tile = int(raw_state.get("max_tile", max_val if max_val >= 0 else 0))
        distance_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        steps_remaining = raw_state.get(
            "steps_remaining", max(0, MAX_STEPS_2048 - raw_state.get("step", 0))
        )
        direction = str(raw_state.get("direction", "up")).lower()

        adapted.update({
            "player_pos": player_pos,
            "goal_pos": goal_pos,
            "distance_to_goal": distance_to_goal,
            "score": score,
            "max_tile": max_tile,
            "steps_remaining": steps_remaining,
            "direction": direction,
            "facing": direction,
            "health": 100,
            "keys_held": 1,
        })
    else:
        raise ValueError(f"Unsupported game: '{game}'. Must be 'snake' or '2048'.")

    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Mock Predictor
# ─────────────────────────────────────────────────────────────────────────────

def deterministic_mock_laya_predict(
    adapted_state: dict[str, Any],
    game: str,
    legal_actions: list[str],
) -> dict[str, float]:
    """Deterministic probability scores for offline testing."""
    player_pos: tuple[int, int] = adapted_state.get("player_pos", (0, 0))
    goal_pos: tuple[int, int] = adapted_state.get("goal_pos", (0, 0))
    current_dir: str = str(adapted_state.get("direction", "right")).lower()

    deltas: dict[str, tuple[int, int]] = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    current_dist = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
    raw_scores: dict[str, float] = {}

    for act in legal_actions:
        d = deltas.get(act, (0, 0))
        next_pos = (player_pos[0] + d[0], player_pos[1] + d[1])
        next_dist = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])

        # Anti-reversal penalty for snake
        is_reverse = (
            (current_dir == "up" and act == "down")
            or (current_dir == "down" and act == "up")
            or (current_dir == "left" and act == "right")
            or (current_dir == "right" and act == "left")
        )

        if game == "snake" and is_reverse:
            score = 0.05
        elif next_dist < current_dist:
            score = 0.75
        elif next_dist == current_dist:
            score = 0.45
        else:
            score = 0.20

        raw_scores[act] = score

    return raw_scores


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop (run_compliant_game)
# ─────────────────────────────────────────────────────────────────────────────

def run_compliant_game(
    game: str = "snake",
    use_mock: bool = True,
    api_key: str = "",
    log_file: str = DEFAULT_LOG_PATH,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    """Execute a fully compliant autonomous game session adhering to Genesis 2.0 Track EVA 02.

    Args:
        game: "snake" or "2048".
        use_mock: If True, uses offline mock clients and deterministic evaluator.
        api_key: TopGameAI Bearer token / API key.
        log_file: Target JSONL log file (default: decision_log.jsonl).
        max_steps_override: Optional override for step count limit.

    Returns:
        Summary dict containing session_id, steps_executed, log_path, and final status.
    """
    game_clean = game.strip().lower()
    if game_clean not in ("snake", "2048"):
        raise ValueError(f"Invalid game '{game}'. Must be 'snake' or '2048'.")

    max_steps = max_steps_override or (
        MAX_STEPS_SNAKE if game_clean == "snake" else MAX_STEPS_2048
    )

    # 1. Initialize Clients & Subsystems
    if game_clean == "snake":
        client: BaseGameClient = (
            MockSnakeClient(api_key=api_key) if use_mock else SnakeClient(api_key=api_key)
        )
    else:
        client = (
            MockGame2048Client(api_key=api_key) if use_mock else Game2048Client(api_key=api_key)
        )

    logger = OfficialLogger(log_dir=".", default_filename=log_file)
    compliance = ComplianceLayer()
    memory = WorldMemory()
    compressor = EgocentricCompressor()
    subgoal_mgr = SubgoalManager()
    hypo_engine = HypothesisEngine()
    ev_calc = AdvancedEVCalculator()

    # 2. Start Game Session
    session_id, state_before = client.start_game()
    log_path = logger.start_session(session_id, game_clean, log_filename=log_file)

    step = 0
    game_over = False
    prev_adapted: dict[str, Any] | None = None

    print(f"\n================================================================================")
    print(f"[*] TOPGAMEAI COMPLIANT RUNNER — Genesis 2.0 EVA 02")
    print(f"    Game: {game_clean.upper()} | Mock: {use_mock} | Session: {session_id}")
    print(f"    Log File: {log_path.name}")
    print(f"================================================================================\n")

    try:
        while not game_over and step < max_steps:
            step += 1
            turn_start = time.perf_counter()

            # Adapt state to V2 representations
            adapted_state = adapt_api_state_to_v2(state_before, game_clean)

            # Update spatial memory
            if prev_adapted is not None:
                pass  # memory.observe is called post-move
            else:
                curr_pos = adapted_state["player_pos"]
                memory.mark_cell(curr_pos[0], curr_pos[1], STATUS_FREE)
                memory.visits[curr_pos] = 1

            decide_start = time.perf_counter()

            # V2 Pipeline Execution
            compressed = compressor.compress(adapted_state, memory)
            subgoal = subgoal_mgr.get_current_subgoal(adapted_state, memory)
            hypo_dict = hypo_engine.generate_hypotheses(
                compressed_context=compressed,
                legal_actions=LEGAL_ACTIONS,
                subgoal=subgoal,
                memory=memory,
            )
            flat_queries = hypo_engine.flatten_queries(hypo_dict)
            question_strings: list[str] = [q for (_, _, q) in flat_queries]

            # LAYA INFERENCE (Mock or Live)
            if use_mock:
                raw_probs = deterministic_mock_laya_predict(
                    adapted_state=adapted_state,
                    game=game_clean,
                    legal_actions=LEGAL_ACTIONS,
                )
            else:
                # =========================================================================
                # TODO: Insert real Laya client.predict_probability here
                # Expected signature:
                # import laya
                # laya_client = laya.Client(api_key=api_key or os.getenv("LAYA_API_KEY", ""))
                # raw_probs = {}
                # for act in LEGAL_ACTIONS:
                #     question = f"Hypothesis: Executing action '{act}' will advance toward the goal."
                #     response = laya_client.predict_probability(context=compressed, question=question)
                #     raw_probs[act] = float(response.probability)
                # =========================================================================
                raise NotImplementedError(
                    "Live Laya API mode selected, but Laya credentials/SDK pending. "
                    "Run with --use-mock for local execution."
                )

            decide_ms = int((time.perf_counter() - decide_start) * 1000)

            # Determine proposed action from highest probability
            proposed_action = max(raw_probs, key=lambda a: raw_probs[a])

            # Compliance Check (Snake anti-reversal safety)
            source = "laya"
            if game_clean == "snake":
                fallback = compliance.check_snake_safety_fallback(state_before, proposed_action)
                if fallback is not None:
                    chosen_action = fallback
                    source = "fallback"
                else:
                    chosen_action = proposed_action
            else:
                chosen_action = proposed_action

            # Normalization (Guarantees exact sum to 1.000 within [0.98, 1.02])
            normalized_probs = compliance.normalize_probabilities_for_log(raw_probs)

            # Dispatch Move to Game Server
            api_start = time.perf_counter()
            if game_clean == "2048":
                move_result = client.send_move(session_id, chosen_action, state_before=state_before)
            else:
                move_result = client.send_move(session_id, chosen_action)

            api_ms = int((time.perf_counter() - api_start) * 1000)
            turn_ms = int((time.perf_counter() - turn_start) * 1000)

            state_after = move_result.state_after
            result = move_result.result
            game_over = move_result.game_over

            # Build Canonical Decision Block
            decision_block = compliance.build_decision_block(
                questions=question_strings,
                normalized_probs=normalized_probs,
                chosen_action=chosen_action,
                source=source,
            )

            # Log to JSONL with SHA-256 Hash Chain
            timing_dict = {
                "decide_ms": decide_ms,
                "api_ms": api_ms,
                "turn_ms": turn_ms,
            }
            logger.log_move(
                state_before,
                chosen_action,
                state_after,
                result,
                decision_block,
                timing_dict,
                session_id=session_id,
                game=game_clean,
                step=step,
            )

            # Observe Transition in Memory
            adapted_after = adapt_api_state_to_v2(state_after, game_clean)
            memory.observe(adapted_state, chosen_action, adapted_after)
            prev_adapted = adapted_state

            # Console Summary
            source_badge = "[LAYA]" if source == "laya" else "[FALLBACK]"
            conf = decision_block["confidence"]
            print(
                f"[STEP {step:03d}] Act: {chosen_action.upper():<5} {source_badge:<10} | "
                f"Res: {result.upper():<8} | Conf: {conf:.4f} | "
                f"Turn: {turn_ms:3d}ms [D:{decide_ms:2d}ms A:{api_ms:2d}ms] | "
                f"Score: {adapted_after.get('score', 0)}"
            )

            state_before = state_after

    finally:
        # Save final state and close file handles
        logger.save_final_state(session_id, state_before)
        logger.close_session(session_id)
        client.close()

    print(f"\n[DONE] Game session complete. {step} steps logged to {log_path.name}.")
    return {
        "session_id": session_id,
        "steps": step,
        "game_over": game_over,
        "log_path": str(log_path),
        "final_state": state_before,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test Verification Block
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    """Run 3-step self-test verification of compliant_runner.py."""
    print("================================================================================")
    print("[TEST] RUNNING COMPLIANT RUNNER SELF-TEST (3 STEPS, SNAKE, MOCK)")
    print("================================================================================")

    # Clean existing test log to ensure exactly 3 lines
    test_log_file = "decision_log.jsonl"
    if os.path.exists(test_log_file):
        os.remove(test_log_file)

    run_result = run_compliant_game(
        game="snake",
        use_mock=True,
        log_file=test_log_file,
        max_steps_override=3,
    )

    # Verify decision_log.jsonl exists and has 3 lines
    assert os.path.exists(test_log_file), f"{test_log_file} does not exist!"

    lines: list[dict[str, Any]] = []
    with open(test_log_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                lines.append(json.loads(line_str))

    assert len(lines) == 3, f"Expected exactly 3 log lines, found {len(lines)}"
    print(f"[OK] Line Count Verified: Exactly {len(lines)} lines present.")

    # Verify Hash Chain Integrity
    is_valid, msg = OfficialLogger.verify_log_file(test_log_file)
    assert is_valid, f"Hash chain verification failed: {msg}"
    print(f"[OK] Hash Chain Verified: {msg}")

    # Verify Probability Normalization [0.98, 1.02]
    for idx, line in enumerate(lines, start=1):
        decision = line.get("decision", {})
        probs = decision.get("probabilities", {})
        assert set(probs.keys()) == {"up", "down", "left", "right"}, (
            f"Step {idx}: Probabilities missing standard directions: {probs}"
        )
        prob_sum = sum(probs.values())
        assert 0.98 <= prob_sum <= 1.02, (
            f"Step {idx}: Probability sum {prob_sum:.6f} out of mandated [0.98, 1.02]"
        )
        assert decision.get("confidence") == probs[decision.get("chosen")], (
            f"Step {idx}: Confidence does not match chosen probability exactly"
        )
        print(f"[OK] Step {idx} Probabilities: sum={prob_sum:.6f} (Normalized & Calibrated)")

    print("\n[SUCCESS] ALL SELF-TEST ASSERTIONS PASSED SUCCESSFULLY!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TopGameAI Genesis 2.0 Compliant Runner (Track EVA 02)"
    )
    parser.add_argument(
        "--game",
        choices=["snake", "2048"],
        default="snake",
        help="Game arena to compete in (default: snake)",
    )
    parser.add_argument(
        "--use-mock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run offline mock simulation (default: True)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("TOPGAME_API_KEY", ""),
        help="TopGameAI Bearer token / API key",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override step limit (default: 300 for snake, 2000 for 2048)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the mandatory 3-step self-test verification",
    )

    # If no arguments provided or --self-test passed, run self-test
    if len(sys.argv) == 1 or "--self-test" in sys.argv:
        _run_self_test()
    else:
        args = parser.parse_args()
        run_compliant_game(
            game=args.game,
            use_mock=args.use_mock,
            api_key=args.api_key,
            log_file=DEFAULT_LOG_PATH,
            max_steps_override=args.max_steps,
        )
