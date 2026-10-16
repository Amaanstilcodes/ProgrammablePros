"""Unit tests for 2048 Decision Brain — Genesis 2.0 Track EVA 02 / Phase 2b.

Tests T1 through T11:
- T1: points_2048(512, 0) == 429
- T2: points_2048(1024, 2500) == 574
- T3: points_2048(8192, 99999) == 1000
- T4: points_2048(64, 5000) == 5
- T5: corner_chain_score == 1.0 on a perfect descending snake board.
- T6: EV divergence proof for 2048: p_progress favors 'right' but p_structure favors 'left' -> selects 'left'.
- T7: A no-op direction is never chosen with source="laya" when a legal alternative exists.
- T8: Fallback budget enforcement blocks the 4th fallback in 30-move session and matches Snake's.
- T9: AST inspection proving no search/planner/minimax/expectimax/RL modules.
- T10: Context length <= 400 characters on 200 randomized boards.
- T11: Rejection does not advance stall_turns or empty_count_history.
- Bonus: StageGate blocks 2048 without unlock confirmation.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from core.compliance_layer import FallbackBudget, points_2048
from core.stage_gate import StageGate, StageLockedError
from games.game2048.features import (
    Board2048Features,
    Game2048FeatureExtractor,
    compute_corner_chain_score,
    format_context_string,
    is_board_changing,
)
from games.game2048.hypotheses import Board2048HypothesisBuilder, HypothesisMap2048
from games.game2048.memory import Board2048Memory
from games.game2048.policy import Board2048Policy
from games.game2048.subgoals import Board2048SubgoalArbiter, Subgoal2048


# ─────────────────────────────────────────────────────────────────────────────
# T1 - T4: Official Score Computation Formula
# ─────────────────────────────────────────────────────────────────────────────

def test_t1_points_512() -> None:
    """T1: points_2048(512, 0) == 429."""
    assert points_2048(512, 0) == 429


def test_t2_points_1024() -> None:
    """T2: points_2048(1024, 2500) == 574."""
    assert points_2048(1024, 2500) == 574


def test_t3_points_8192_cap() -> None:
    """T3: points_2048(8192, 99999) == 1000."""
    assert points_2048(8192, 99999) == 1000


def test_t4_points_64() -> None:
    """T4: points_2048(64, 5000) == 5."""
    assert points_2048(64, 5000) == 5


# ─────────────────────────────────────────────────────────────────────────────
# T5: Corner Chain Score on Perfect Descending Board
# ─────────────────────────────────────────────────────────────────────────────

def test_t5_perfect_descending_snake_chain() -> None:
    """T5: corner_chain_score == 1.0 on a perfect descending snake board."""
    # Anchored at (3, 0) bottom-left:
    # Row 3: 2048, 1024, 512, 256
    # Row 2:   16,   32,  64, 128
    # Row 1:    8,    4,   2,   0
    # Row 0:    0,    0,   0,   0
    perfect_board = (
        (0, 0, 0, 0),
        (8, 4, 2, 0),
        (16, 32, 64, 128),
        (2048, 1024, 512, 256),
    )
    score = compute_corner_chain_score(perfect_board)
    assert score == 1.0, f"Expected perfect chain score 1.0, got {score}"


# ─────────────────────────────────────────────────────────────────────────────
# T6: EV Divergence Proof (Progress vs Structure)
# ─────────────────────────────────────────────────────────────────────────────

def test_t6_ev_divergence_proof() -> None:
    """T6: p_progress favors 'right' but p_structure favors 'left' -> selects 'left'."""
    policy = Board2048Policy(mode="full")
    # Board with 512 at bottom-left (3, 0) and horizontal merge available in row 0
    board = (
        (2, 2, 8, 16),
        (4, 8, 16, 32),
        (8, 16, 32, 64),
        (512, 256, 128, 64),
    )
    extractor = Game2048FeatureExtractor()
    features = extractor.extract_features({"board": board, "score": 2000})
    subgoal = Subgoal2048(
        name="KEEP_CORNER",
        goal_phrase="keeping the largest tile fixed in the bottom-left corner",
        priority=3,
        rationale="EV proof test",
    )

    hypo_map = HypothesisMap2048(
        mapping={
            ("up", "progress"): 0, ("up", "danger"): 1, ("up", "structure"): 2,
            ("down", "progress"): 3, ("down", "danger"): 4, ("down", "structure"): 5,
            ("left", "progress"): 6, ("left", "danger"): 7, ("left", "structure"): 8,
            ("right", "progress"): 9, ("right", "danger"): 10, ("right", "structure"): 11,
        }
    )
    # Right has higher progress (0.8 vs 0.3) but destroys structure (0.0 vs 0.95)
    # and incurs corner_break_penalty (+4.0)
    outputs = [
        0.0, 1.0, 0.0,   # up
        0.0, 1.0, 0.0,   # down
        0.3, 0.0, 0.95,  # left: modest progress, high structure, no corner penalty
        0.8, 0.0, 0.05,  # right: high progress, terrible structure, corner break penalty (+4.0)
    ]
    decision = policy.decide(features, subgoal, hypo_map, outputs)

    assert decision.action == "left", (
        f"Expected 'left' to win via structure payoff and corner protection, got '{decision.action}'"
    )
    assert decision.source == "laya"


# ─────────────────────────────────────────────────────────────────────────────
# T7: No-Op Direction Never Chosen with source="laya"
# ─────────────────────────────────────────────────────────────────────────────

def test_t7_no_op_direction_never_chosen_with_laya() -> None:
    """T7: A no-op direction is never chosen with source='laya' when a legal alternative exists."""
    policy = Board2048Policy(mode="full")
    # All tiles compressed to the bottom; sliding 'down' changes nothing
    board = (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (2, 4, 8, 16),
        (4, 8, 16, 32),
    )
    assert not is_board_changing(board, "down"), "Down must be a no-op on this board"
    assert is_board_changing(board, "up"), "Up must be legal"

    extractor = Game2048FeatureExtractor()
    features = extractor.extract_features({"board": board, "score": 500})
    subgoal = Subgoal2048("KEEP_CORNER", "test phrase", 3, "test")

    hypo_map = HypothesisMap2048(
        mapping={
            ("up", "progress"): 0, ("up", "danger"): 1, ("up", "structure"): 2,
            ("down", "progress"): 3, ("down", "danger"): 4, ("down", "structure"): 5,
            ("left", "progress"): 6, ("left", "danger"): 7, ("left", "structure"): 8,
            ("right", "progress"): 9, ("right", "danger"): 10, ("right", "structure"): 11,
        }
    )
    # Down is given 1.0 progress, but illegal penalty (-100) must disqualify it
    outputs = [
        0.4, 0.0, 0.5,  # up (legal)
        1.0, 0.0, 0.5,  # down (no-op!)
        0.0, 1.0, 0.0,  # left
        0.0, 1.0, 0.0,  # right
    ]
    decision = policy.decide(features, subgoal, hypo_map, outputs)

    assert decision.action != "down", "No-op direction must not be selected!"
    assert decision.action == "up"
    assert decision.source == "laya"


# ─────────────────────────────────────────────────────────────────────────────
# T8: Fallback Budget Parity with Snake
# ─────────────────────────────────────────────────────────────────────────────

def test_t8_fallback_budget_parity() -> None:
    """T8: Fallback budget blocks 4th fallback in a 30-move session."""
    budget = FallbackBudget(max_ratio=0.10)
    accepted = 30
    assert budget.may_fallback(accepted, 0) is True
    assert budget.may_fallback(accepted, 1) is True
    assert budget.may_fallback(accepted, 2) is True
    assert budget.may_fallback(accepted, 3) is False  # 4th fallback blocked


# ─────────────────────────────────────────────────────────────────────────────
# T9: Zero Search/Planner Module AST Inspection
# ─────────────────────────────────────────────────────────────────────────────

def test_t9_no_search_or_planner_modules() -> None:
    """T9: AST inspection proving no minimax, expectimax, MCTS, BFS, or pathfinding in game2048."""
    forbidden = {
        "minimax", "expectimax", "mcts", "bfs", "dfs", "astar", "a_star",
        "scipy", "networkx", "torch", "tensorflow", "gym", "gymnasium",
    }
    game_dir = Path("games/game2048")
    assert game_dir.is_dir()

    py_files = list(game_dir.glob("*.py"))
    assert len(py_files) >= 5

    for fpath in py_files:
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.lower().split(".")
                    assert not any(p in forbidden for p in parts), (
                        f"Forbidden import '{alias.name}' in {fpath.name}!"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = node.module.lower().split(".")
                    assert not any(p in forbidden for p in parts), (
                        f"Forbidden import from '{node.module}' in {fpath.name}!"
                    )
            elif isinstance(node, ast.Call):
                fid = ""
                if isinstance(node.func, ast.Name):
                    fid = node.func.id.lower()
                elif isinstance(node.func, ast.Attribute):
                    fid = node.func.attr.lower()
                assert fid not in forbidden, f"Forbidden call '{fid}' in {fpath.name}!"


# ─────────────────────────────────────────────────────────────────────────────
# T10: Context String Length Bounded on 200 Randomized Boards
# ─────────────────────────────────────────────────────────────────────────────

def test_t10_context_string_length_bounded() -> None:
    """T10: Context length <= 400 characters on 200 randomized boards."""
    rng = random.Random(1337)
    tile_pool = [0, 0, 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    subgoal_names = ["SURVIVE", "RESTORE_CORNER", "KEEP_CORNER"]
    extractor = Game2048FeatureExtractor()
    memory = Board2048Memory()

    for _ in range(200):
        board = tuple(
            tuple(rng.choice(tile_pool) for _ in range(4))
            for _ in range(4)
        )
        subgoal = rng.choice(subgoal_names)
        features = extractor.extract_features({"board": board, "score": rng.randint(0, 50000)}, memory)
        ctx = format_context_string(features, subgoal_name=subgoal)

        assert len(ctx) <= 400, f"Context string exceeded 400 chars ({len(ctx)}): {ctx}"
        assert len(ctx) > 30, f"Context string too short: {ctx}"
        assert "game=2048" in ctx
        assert f"max={features.max_tile}" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# T11: Rejection Invariant Preservation
# ─────────────────────────────────────────────────────────────────────────────

def test_t11_rejection_does_not_advance_stall_or_empty_history() -> None:
    """T11: Rejection does not advance stall_turns or empty_count_history."""
    memory = Board2048Memory()
    board = ((2, 4, 8, 16), (4, 8, 16, 32), (0, 0, 0, 0), (0, 0, 0, 0))
    state_before = {"board": board, "score": 100, "best_tile": 32}

    # Step 1: applied move
    memory.observe_transition(state_before, "down", state_before, "applied")
    stall_before = memory.stall_turns
    empty_hist_len_before = len(memory.empty_count_history)

    # Step 2: rejected move
    memory.observe_transition(state_before, "down", state_before, "rejected")

    assert memory.stall_turns == stall_before, "stall_turns must not increment on rejection!"
    assert len(memory.empty_count_history) == empty_hist_len_before, (
        "empty_count_history must not be appended on rejection!"
    )
    assert len(memory.rejection_history) == 1
    assert memory.rejection_history[0][0] == "down"


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: StageGate Enforces Stage 2 Live Lock
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_gate_refuses_live_2048_without_unlock() -> None:
    """StageGate refuses to run 2048 live when qualification record is absent."""
    gate = StageGate(log_root="logs_nonexistent", unlock_marker_dir="submission/nonexistent")
    with pytest.raises(StageLockedError):
        gate.assert_stage_unlocked("2048", allow_override=False)
