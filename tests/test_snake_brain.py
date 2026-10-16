"""Unit tests for Snake Decision Brain — Genesis 2.0 Track EVA 02 / Phase 2a.

Tests T1 through T7:
- T1: Reversal is never chosen with source="laya" when a legal alternative exists and has higher progress probability.
- T2: A wall-adjacent direction receives EV < 0 even with p_progress=0.95.
- T3: Policy does NOT choose the highest-progress action when p_danger is high (EV vs argmax proof).
- T4: Fallback budget blocks the 4th fallback in a 30-move session.
- T5: When budget is exhausted, the re-query path is used and source remains "laya".
- T6: AST inspection proving no search/planner/BFS/A* imports or calls in games/snake.
- T7: Context string length <= 400 on 200 randomized synthetic states.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from core.compliance_layer import FallbackBudget
from games.snake.features import FeatureExtractor, SnakeFeatures, format_context_string
from games.snake.hypotheses import HypothesisMap, SnakeHypothesisBuilder
from games.snake.memory import SnakeMemory
from games.snake.policy import SnakeDecision, SnakePolicy
from games.snake.subgoals import SnakeSubgoalArbiter, Subgoal


def _build_synthetic_features(
    head: tuple[int, int] = (5, 5),
    direction: str = "right",
    length: int = 4,
    food: tuple[int, int] | None = (8, 5),
    board_w: int = 16,
    board_h: int = 16,
    stall: int = 0,
    warning: str = "",
    cramped: bool = False,
    body_ahead: bool = False,
    wall_ahead: bool = False,
    occupied_fraction: float = 0.1,
) -> SnakeFeatures:
    """Helper to synthesize test features."""
    free_neighbors = {
        "up": True,
        "down": True,
        "left": False if direction == "right" else True,
        "right": not (wall_ahead or body_ahead),
    }
    dist_wall = {
        "up": head[1],
        "down": (board_h - 1) - head[1],
        "left": head[0],
        "right": (board_w - 1) - head[0],
    }
    dx = (food[0] - head[0]) if food else None
    dy = (food[1] - head[1]) if food else None
    dist = (abs(dx) + abs(dy)) if (dx is not None and dy is not None) else None
    bearing = "E" if (food and food[0] > head[0]) else "HERE"

    return SnakeFeatures(
        head=head,
        direction=direction,
        body_length=length,
        board_w=board_w,
        board_h=board_h,
        free_neighbors=free_neighbors,
        body_ahead=body_ahead,
        body_left=False,
        body_right=False,
        wall_ahead=wall_ahead,
        wall_left=False,
        wall_right=False,
        distance_to_wall=dist_wall,
        food=food,
        food_delta=(dx, dy) if (dx is not None and dy is not None) else None,
        food_manhattan=dist,
        food_bearing=bearing if food else None,
        food_axis_aligned=(dx == 0 or dy == 0) if food else False,
        neck=(head[0] - 1, head[1]),
        tail=(head[0] - (length - 1), head[1]),
        tail_manhattan=length - 1,
        body_occupies_ahead_2=False,
        occupied_fraction=occupied_fraction,
        visit_count_at_head=1,
        recent_actions=(direction,),
        recent_failure_warning=warning,
        food_stall_turns=stall,
        cramped=cramped,
    )


def test_t1_reversal_never_chosen_with_laya() -> None:
    """T1: Reversal is never chosen with source="laya" when a legal alternative exists."""
    policy = SnakePolicy(mode="full")
    features = _build_synthetic_features(
        head=(5, 5),
        direction="up",  # Reversal is "down"
        food=(5, 8),     # Food is in down direction
    )
    subgoal = Subgoal(
        name="CHASE_FOOD",
        goal_phrase="reaching the food while keeping an escape route open",
        priority=3,
        rationale="Test",
    )
    # Order: [up.p, up.d, up.s, down.p, down.d, down.s, left.p, left.d, left.s, right.p, right.d, right.s]
    hypo_map = HypothesisMap(
        mapping={
            ("up", "progress"): 0, ("up", "danger"): 1, ("up", "space"): 2,
            ("down", "progress"): 3, ("down", "danger"): 4, ("down", "space"): 5,
            ("left", "progress"): 6, ("left", "danger"): 7, ("left", "space"): 8,
            ("right", "progress"): 9, ("right", "danger"): 10, ("right", "space"): 11,
        }
    )
    # Give "down" (reversal) high progress probability, but "right" is legal with positive progress
    outputs = [
        0.2, 0.0, 0.5,   # up
        0.9, 0.0, 0.5,   # down (reversal)
        0.1, 0.0, 0.5,   # left
        0.8, 0.0, 0.5,   # right (legal alternative)
    ]
    decision = policy.decide(features, subgoal, hypo_map, outputs)

    assert decision.action != "down", "Reversal must never be chosen!"
    assert decision.action == "right"
    assert decision.source == "laya"


def test_t2_wall_adjacent_direction_ev_negative() -> None:
    """T2: A wall-adjacent direction receives EV < 0 even with p_progress=0.95."""
    policy = SnakePolicy(mode="full")
    # Head at (0, 5), so "left" immediately hits the board boundary
    features = _build_synthetic_features(head=(0, 5), direction="up")
    # Force free_neighbors["left"] = False because it's into the wall
    features_dict = dict(features.__dict__)
    features_dict["free_neighbors"] = {"up": True, "down": False, "left": False, "right": True}
    features_wall = SnakeFeatures(**features_dict)

    subgoal = Subgoal("CHASE_FOOD", "test phrase", 3, "test")
    hypo_map = HypothesisMap(
        mapping={
            ("up", "progress"): 0, ("up", "danger"): 1, ("up", "space"): 2,
            ("down", "progress"): 3, ("down", "danger"): 4, ("down", "space"): 5,
            ("left", "progress"): 6, ("left", "danger"): 7, ("left", "space"): 8,
            ("right", "progress"): 9, ("right", "danger"): 10, ("right", "space"): 11,
        }
    )
    outputs = [
        0.5, 0.0, 0.5,  # up
        0.0, 0.0, 0.5,  # down
        0.95, 0.0, 0.5, # left (wall-adjacent, high progress)
        0.5, 0.0, 0.5,  # right
    ]
    decision = policy.decide(features_wall, subgoal, hypo_map, outputs)

    left_ev = decision.ev_breakdown["left"]["ev"]
    assert left_ev < 0.0, f"Expected EV < 0 for wall direction, got {left_ev}"


def test_t3_ev_vs_argmax_proof() -> None:
    """T3: High progress + high danger does NOT win over lower progress + safe (EV vs argmax proof)."""
    policy = SnakePolicy(mode="full")
    features = _build_synthetic_features(head=(5, 5), direction="right")
    subgoal = Subgoal("CHASE_FOOD", "test phrase", 3, "test")

    hypo_map = HypothesisMap(
        mapping={
            ("up", "progress"): 0, ("up", "danger"): 1, ("up", "space"): 2,
            ("down", "progress"): 3, ("down", "danger"): 4, ("down", "space"): 5,
            ("left", "progress"): 6, ("left", "danger"): 7, ("left", "space"): 8,
            ("right", "progress"): 9, ("right", "danger"): 10, ("right", "space"): 11,
        }
    )
    # "right": progress = 0.9, danger = 0.8  -> EV: 0.9*10 - 0.8*40 + 1.5 - 0.2 = -21.7
    # "up":    progress = 0.1, danger = 0.01 -> EV: 0.1*10 - 0.01*40 + 1.5 - 1.8 = 0.3
    outputs = [
        0.1, 0.01, 0.5,   # up
        0.05, 0.05, 0.5,  # down
        0.05, 0.05, 0.5,  # left
        0.90, 0.80, 0.5,  # right (highest progress, but dangerous)
    ]
    decision = policy.decide(features, subgoal, hypo_map, outputs)

    assert decision.action != "right", "Policy must NOT choose highest-progress action when danger is high!"
    assert decision.action == "up"
    assert decision.source == "laya"


def test_t4_fallback_budget_blocks_fourth_fallback() -> None:
    """T4: Fallback budget blocks the 4th fallback in a 30-move session."""
    budget = FallbackBudget(max_ratio=0.10)
    accepted_so_far = 30

    # Fallback 1 (fallbacks_so_far = 0): (0+1) <= 0.10 * 31 = 3.1 -> True
    assert budget.may_fallback(accepted_so_far, 0) is True
    # Fallback 2 (fallbacks_so_far = 1): (1+1) <= 3.1 -> True
    assert budget.may_fallback(accepted_so_far, 1) is True
    # Fallback 3 (fallbacks_so_far = 2): (2+1) <= 3.1 -> True
    assert budget.may_fallback(accepted_so_far, 2) is True
    # Fallback 4 (fallbacks_so_far = 3): (3+1) <= 3.1 -> False (BLOCKED!)
    assert budget.may_fallback(accepted_so_far, 3) is False


def test_t5_budget_exhausted_uses_requery_and_source_stays_laya() -> None:
    """T5: When budget is exhausted and chosen action is illegal, re-query path keeps source='laya'."""
    policy = SnakePolicy(mode="full")
    features = _build_synthetic_features(head=(5, 5), direction="up")
    subgoal = Subgoal("CHASE_FOOD", "test phrase", 3, "test")

    # When all moves are marked illegal (e.g. cramped pocket), 'up' has higher EV than 'right' initially
    features_dict = dict(features.__dict__)
    features_dict["free_neighbors"] = {"up": False, "down": False, "left": False, "right": False}
    features_cramped = SnakeFeatures(**features_dict)

    hypo_map = HypothesisMap(
        mapping={
            ("up", "progress"): 0, ("up", "danger"): 1, ("up", "space"): 2,
            ("down", "progress"): 3, ("down", "danger"): 4, ("down", "space"): 5,
            ("left", "progress"): 6, ("left", "danger"): 7, ("left", "space"): 8,
            ("right", "progress"): 9, ("right", "danger"): 10, ("right", "space"): 11,
        }
    )
    # Output that gives "up" higher EV than "right" (-88.5 vs -96.9), causing "up" to be chosen initially
    outputs = [
        1.0, 0.0, 0.5,  # up (highest progress, selected first by EV)
        0.0, 1.0, 0.5,  # down
        0.0, 1.0, 0.5,  # left
        0.3, 0.0, 0.5,  # right (second highest EV)
    ]
    # Budget exhausted: 30 accepted, 3 fallbacks -> may_fallback(30, 3) is False
    decision = policy.decide(
        features=features_cramped,
        subgoal=subgoal,
        hypothesis_map=hypo_map,
        laya_outputs=outputs,
        accepted_so_far=30,
        fallbacks_so_far=3,
    )

    assert decision.source == "laya", f"Source must stay 'laya', got {decision.source}"
    assert decision.action == "right", f"Expected 'right' after grounding 'up', got {decision.action}"
    assert decision.fallback_reason is not None
    assert "Budget exhausted" in decision.fallback_reason


def test_t6_no_search_or_pathfinding_modules() -> None:
    """T6: AST inspection proving no BFS, DFS, A*, flood-fill, or pathfinding imports/calls."""
    forbidden_tokens = {
        "bfs", "dfs", "astar", "a_star", "flood_fill", "floodfill",
        "dijkstra", "pathfinding", "networkx", "scipy", "heapq",
    }
    snake_dir = Path("games/snake")
    assert snake_dir.is_dir()

    py_files = list(snake_dir.glob("*.py"))
    assert len(py_files) >= 5, "Expected all 5 Snake brain files to exist."

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name_parts = alias.name.lower().split(".")
                    assert not any(part in forbidden_tokens for part in name_parts), (
                        f"Forbidden import '{alias.name}' detected in {py_file.name}!"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_parts = node.module.lower().split(".")
                    assert not any(part in forbidden_tokens for part in mod_parts), (
                        f"Forbidden import from '{node.module}' detected in {py_file.name}!"
                    )
            elif isinstance(node, ast.Call):
                func_id = ""
                if isinstance(node.func, ast.Name):
                    func_id = node.func.id.lower()
                elif isinstance(node.func, ast.Attribute):
                    func_id = node.func.attr.lower()
                assert func_id not in forbidden_tokens, (
                    f"Forbidden function call '{func_id}' detected in {py_file.name}!"
                )


def test_t7_context_string_length_bounded() -> None:
    """T7: Context string length <= 400 characters across 200 randomized synthetic states."""
    rng = random.Random(42)
    directions = ["up", "down", "left", "right"]
    subgoal_names = ["SURVIVE", "REGROUP", "CHASE_FOOD"]
    sample_warnings = [
        "",
        "WARNING: 'left' reverses the current direction and is illegal.",
        "WARNING: 'up' was rejected by the game. Do not repeat it.",
        "WARNING: No food eaten for 24 turns. Prefer safer positioning over greedy approach.",
        "WARNING: 'down' reduced open space. Prefer preserving escape room.",
    ]

    for _ in range(200):
        head = (rng.randint(0, 15), rng.randint(0, 15))
        direction = rng.choice(directions)
        length = rng.randint(1, 100)
        has_food = rng.random() > 0.1
        food = (rng.randint(0, 15), rng.randint(0, 15)) if has_food else None
        stall = rng.randint(0, 50)
        warning = rng.choice(sample_warnings)
        occupied = round(rng.uniform(0.01, 0.95), 2)
        subgoal_name = rng.choice(subgoal_names)

        features = _build_synthetic_features(
            head=head,
            direction=direction,
            length=length,
            food=food,
            stall=stall,
            warning=warning,
            occupied_fraction=occupied,
        )
        ctx = format_context_string(features, subgoal_name=subgoal_name)

        assert len(ctx) <= 400, f"Context string exceeded 400 chars ({len(ctx)}): {ctx}"
        assert len(ctx) > 30, f"Context string too short: {ctx}"
        assert "game=snake" in ctx
        assert f"head=({head[0]},{head[1]})" in ctx
        assert f"dir={direction}" in ctx


def test_subgoal_cascade_and_no_action_field() -> None:
    """Assert correct subgoal priority cascade and that Subgoal contains NO action field."""
    arbiter = SnakeSubgoalArbiter()

    # P1 SURVIVE: cramped
    f_cramped = _build_synthetic_features(cramped=True)
    sg1 = arbiter.select(f_cramped)
    assert sg1.name == "SURVIVE"
    assert sg1.priority == 1

    # P1 SURVIVE: wall_ahead
    f_wall = _build_synthetic_features(wall_ahead=True)
    sg1_wall = arbiter.select(f_wall)
    assert sg1_wall.name == "SURVIVE"

    # P2 REGROUP: stall >= 8 and occupied > 0.35
    f_regroup = _build_synthetic_features(cramped=False, stall=9, occupied_fraction=0.40)
    sg2 = arbiter.select(f_regroup)
    assert sg2.name == "REGROUP"
    assert sg2.priority == 2

    # P3 CHASE_FOOD: normal state
    f_food = _build_synthetic_features(cramped=False, stall=2, occupied_fraction=0.10)
    sg3 = arbiter.select(f_food)
    assert sg3.name == "CHASE_FOOD"
    assert sg3.priority == 3

    # Assert no field in Subgoal is an action or direction
    for sg in [sg1, sg2, sg3]:
        for field_name in ["name", "goal_phrase", "priority", "rationale"]:
            val = str(getattr(sg, field_name)).lower()
            assert val not in ["up", "down", "left", "right"], (
                f"Subgoal field '{field_name}' must NOT contain an action!"
            )
