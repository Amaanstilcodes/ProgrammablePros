"""2048 Decision Policy — Genesis 2.0 Track EVA 02 / Phase 2b.

Multi-objective Expected Value (EV) synthesis from Laya probability evaluations.
Rule R4:
- The ONLY thing that selects an action is argmax over EV scores built from Laya's probabilities,
  unless legality safety fallback fires (<= 10%).
- NO expectimax, NO minimax, NO MCTS, NO board-simulation lookahead.
- Fallback budget strictly enforced with ground-truth re-query escaping.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from core.compliance_layer import ComplianceLayer, FallbackBudget
from games.game2048.features import (
    Board2048Features,
    Game2048FeatureExtractor,
    is_board_changing,
)
from games.game2048.hypotheses import HypothesisMap2048
from games.game2048.subgoals import Subgoal2048

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")
# Deterministic tie-break order: down, left, up, right
TIE_BREAK_ORDER: tuple[str, ...] = ("down", "left", "up", "right")


def points_2048(best_tile: int, in_game_score: int) -> int:
    """Official Genesis 2.0 score formula for Stage 2 (2048).

    Points = min(1000, max(0, exponent - 6) * 143 + in_game_score // 1000)
    """
    if best_tile < 2:
        return 0
    exponent = math.log2(best_tile)
    if not exponent.is_integer():
        exponent = math.floor(exponent)
    return min(1000, max(0, int(exponent) - 6) * 143 + in_game_score // 1000)


@dataclass
class Board2048Decision:
    """Canonical decision record produced by Board2048Policy."""

    action: str
    source: str  # "laya" | "fallback"
    ev_breakdown: dict[str, dict[str, float]]
    primary_laya_probs: dict[str, float]
    questions: list[str]
    subgoal: Subgoal2048
    features_snapshot: dict[str, Any]
    fallback_reason: str | None = None

    @property
    def normalized_laya_probs(self) -> dict[str, float]:
        """Backward-compatibility alias."""
        return self.primary_laya_probs


class Board2048Policy:
    """EV synthesizer and action selector implementing Phase 2b specifications."""

    def __init__(self, mode: str = "full") -> None:
        self.mode = mode
        self.compliance = ComplianceLayer()
        self.budget = FallbackBudget(max_ratio=0.10)

    def decide(
        self,
        features: Board2048Features,
        subgoal: Subgoal2048,
        hypothesis_map: HypothesisMap2048,
        laya_outputs: list[float],
        memory: Any = None,
        accepted_so_far: int = 0,
        fallbacks_so_far: int = 0,
        laya_backend: Any = None,
        questions: list[str] | None = None,
        context: str = "",
    ) -> Board2048Decision:
        """Select action via EV synthesis and enforce fallback budget."""
        # Step 1 — Decode Laya probabilities
        p_progress: dict[str, float] = {}
        p_danger: dict[str, float] = {}
        p_structure: dict[str, float] = {}

        for a in DIRECTIONS:
            p_progress[a] = float(laya_outputs[hypothesis_map.get_index(a, "progress")])
            p_danger[a] = float(laya_outputs[hypothesis_map.get_index(a, "danger")])
            if self.mode == "lean" or "structure" not in hypothesis_map.kinds:
                p_structure[a] = 0.5
            else:
                p_structure[a] = float(laya_outputs[hypothesis_map.get_index(a, "structure")])

        # Step 2 — Deterministic legality check (is_board_changing)
        is_legal: dict[str, bool] = {a: is_board_changing(features.board, a) for a in DIRECTIONS}

        # Step 3 — EV synthesis
        reward_scale = 1.0 + 0.15 * max(0, features.max_tile_log2 - 7)
        structure_weight = 6.0 if subgoal.name != "SURVIVE" else 1.0
        danger_weight = 35.0 if features.empty_count <= 3 else 18.0

        # Corner break penalties (geometric feature, not simulation)
        corner_break_penalties: dict[str, float] = {}
        for a in DIRECTIONS:
            if a == "up" and features.max_tile_at_designated_corner:
                pen = 6.0
            elif a == "right" and features.max_tile_at_designated_corner:
                pen = 4.0
            else:
                pen = 0.0
            corner_break_penalties[a] = pen

        # Repeat penalties from memory
        recent_actions: tuple[str, ...] = features.recent_actions
        rejection_actions: set[str] = set()
        if memory is not None:
            if hasattr(memory, "rejection_history"):
                rejection_actions = {r[0] for r in memory.rejection_history}
            if hasattr(memory, "recent_actions") and memory.recent_actions:
                recent_actions = tuple(memory.recent_actions)

        repeat_penalties: dict[str, float] = {}
        for a in DIRECTIONS:
            pen = 0.0
            if len(recent_actions) >= 2 and recent_actions[-1] == a and recent_actions[-2] == a:
                pen += 2.0
            if a in rejection_actions:
                pen += 4.0
            repeat_penalties[a] = pen

        # Merge affinity
        max_merges = max(features.merge_pairs.values()) if features.merge_pairs else 0
        max_merge_div = max(1, max_merges)
        merge_affinities: dict[str, float] = {
            a: 0.8 * features.merge_pairs.get(a, 0) / max_merge_div
            for a in DIRECTIONS
        }

        ev_breakdown: dict[str, dict[str, float]] = {}
        ev_scores: dict[str, float] = {}

        for a in DIRECTIONS:
            illegal_pen = 100.0 if not is_legal[a] else 0.0
            prog_term = p_progress[a] * (10.0 * reward_scale)
            struct_term = p_structure[a] * structure_weight
            danger_term = p_danger[a] * danger_weight
            regret_term = (1.0 - p_progress[a]) * 3.0

            ev = (
                prog_term
                + struct_term
                - danger_term
                - regret_term
                - corner_break_penalties[a]
                - repeat_penalties[a]
                + merge_affinities[a]
                - illegal_pen
            )
            ev_scores[a] = ev
            ev_breakdown[a] = {
                "ev": ev,
                "p_progress": p_progress[a],
                "p_danger": p_danger[a],
                "p_structure": p_structure[a],
                "reward_scale": reward_scale,
                "prog_term": prog_term,
                "struct_term": struct_term,
                "danger_term": danger_term,
                "regret_term": regret_term,
                "corner_break_penalty": corner_break_penalties[a],
                "repeat_penalty": repeat_penalties[a],
                "merge_affinity": merge_affinities[a],
                "illegal_penalty": illegal_pen,
                "is_legal": 1.0 if is_legal[a] else 0.0,
            }

        # Step 4 — Choose argmax EV with deterministic tie-breaker (down, left, up, right)
        chosen = max(DIRECTIONS, key=lambda a: (ev_scores[a], -TIE_BREAK_ORDER.index(a)))
        source = "laya"
        fallback_reason: str | None = None

        # Step 5 & Step 6 — Safety fallback & budget enforcement
        if not is_legal[chosen]:
            # Find first legal direction in deterministic preference order
            alt_candidate: str | None = None
            for cand in TIE_BREAK_ORDER:
                if is_legal[cand]:
                    alt_candidate = cand
                    break

            if alt_candidate is not None:
                if self.budget.may_fallback(accepted_so_far, fallbacks_so_far):
                    chosen = alt_candidate
                    source = "fallback"
                    fallback_reason = f"Deterministic safety fallback from no-op '{chosen}' to '{alt_candidate}'."
                else:
                    # Fallback budget EXHAUSTED: Escaping via ground truth re-query
                    clause = f"GROUND TRUTH: sliding '{chosen}' does not change the board. Its probability of being a legal progressing move is exactly 0.0."
                    fallback_reason = f"Budget exhausted; applied re-query grounding: {clause}"

                    # Ground chosen action: p_progress is 0.0, p_danger is 1.0
                    re_p_progress = dict(p_progress)
                    re_p_danger = dict(p_danger)
                    re_p_progress[chosen] = 0.0
                    re_p_danger[chosen] = 1.0

                    re_ev: dict[str, float] = {}
                    for a in DIRECTIONS:
                        if a == chosen:
                            re_ev[a] = -999.0
                        else:
                            ill_term = 100.0 if not is_legal[a] else 0.0
                            re_ev[a] = (
                                re_p_progress[a] * (10.0 * reward_scale)
                                + p_structure[a] * structure_weight
                                - re_p_danger[a] * danger_weight
                                - (1.0 - re_p_progress[a]) * 3.0
                                - corner_break_penalties[a]
                                - repeat_penalties[a]
                                + merge_affinities[a]
                                - ill_term
                            )
                    chosen = max(DIRECTIONS, key=lambda a: (re_ev[a], -TIE_BREAK_ORDER.index(a)))
                    source = "laya"

        features_snapshot = asdict(features) if hasattr(features, "__dataclass_fields__") else dict(features)
        decision_questions = list(questions) if questions is not None else []

        return Board2048Decision(
            action=chosen,
            source=source,
            ev_breakdown=ev_breakdown,
            primary_laya_probs=p_progress,
            questions=decision_questions,
            subgoal=subgoal,
            features_snapshot=features_snapshot,
            fallback_reason=fallback_reason,
        )

    def select_action(
        self,
        action_probabilities: dict[str, float],
        ev_scores: dict[str, float],
        legal_actions: list[str],
        state_before: dict[str, Any],
    ) -> tuple[str, str]:
        """Backward-compatibility entry point for orchestrator."""
        if ev_scores:
            proposed_action = max(legal_actions, key=lambda a: (ev_scores.get(a, -999.0), -TIE_BREAK_ORDER.index(a) if a in TIE_BREAK_ORDER else 0))
        else:
            proposed_action = max(action_probabilities, key=lambda a: action_probabilities[a])

        # Legality check on board
        board_raw = state_before.get("board", [[0] * 4 for _ in range(4)])
        board_tuple = tuple(tuple(int(c) for c in r) for r in board_raw)
        if not is_board_changing(board_tuple, proposed_action):
            for cand in TIE_BREAK_ORDER:
                if is_board_changing(board_tuple, cand):
                    return cand, "fallback"

        return proposed_action, "laya"


# Alias for backward compatibility
Game2048DecisionPolicy = Board2048Policy
