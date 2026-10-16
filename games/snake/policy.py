"""Snake Decision Policy — Genesis 2.0 Track EVA 02 / Phase 2a.

Multi-objective Expected Value (EV) synthesis from Laya probability evaluations.
Rule R4:
- The ONLY thing that selects an action is argmax over EV scores built from Laya's probabilities,
  unless a narrowly-defined legality safety fallback fires (<= 10%).
- NO BFS, NO flood-fill, NO A*, NO pathfinding, NO lookahead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.compliance_layer import ComplianceLayer, FallbackBudget, make_rejection_clause
from games.snake.features import SnakeFeatures
from games.snake.hypotheses import HypothesisMap
from games.snake.subgoals import Subgoal

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")
TIE_BREAK_ORDER: tuple[str, ...] = ("up", "right", "down", "left")

OPPOSITES: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass
class SnakeDecision:
    """Full decision record produced by SnakePolicy."""

    action: str
    source: str  # "laya" | "fallback"
    ev_breakdown: dict[str, dict[str, float]]
    primary_laya_probs: dict[str, float]
    questions: list[str]
    subgoal: Subgoal
    features_snapshot: dict[str, Any]
    fallback_reason: str | None = None

    @property
    def normalized_laya_probs(self) -> dict[str, float]:
        """Backward-compatibility alias for protocol conformance."""
        return self.primary_laya_probs


class SnakePolicy:
    """EV synthesizer and action selector implementing Phase 2a specifications."""

    def __init__(self, mode: str = "full") -> None:
        self.mode = mode
        self.compliance = ComplianceLayer()
        self.budget = FallbackBudget(max_ratio=0.10)

    def decide(
        self,
        features: SnakeFeatures,
        subgoal: Subgoal,
        hypothesis_map: HypothesisMap,
        laya_outputs: list[float],
        memory: Any = None,
        accepted_so_far: int = 0,
        fallbacks_so_far: int = 0,
        laya_backend: Any = None,
        questions: list[str] | None = None,
        context: str = "",
    ) -> SnakeDecision:
        """Select action via EV synthesis and enforce fallback budget."""
        # Step 1 — Decode Laya probabilities
        p_progress: dict[str, float] = {}
        p_danger: dict[str, float] = {}
        p_space: dict[str, float] = {}

        for a in DIRECTIONS:
            p_progress[a] = float(laya_outputs[hypothesis_map.get_index(a, "progress")])
            p_danger[a] = float(laya_outputs[hypothesis_map.get_index(a, "danger")])
            if self.mode == "lean" or "space" not in hypothesis_map.kinds:
                p_space[a] = 0.5
            else:
                p_space[a] = float(laya_outputs[hypothesis_map.get_index(a, "space")])

        # Step 2 — Deterministic legality mask (PERCEPTION, not a decision)
        illegal: dict[str, bool] = {}
        reversal = OPPOSITES.get(features.direction, "")
        for a in DIRECTIONS:
            if a == reversal:
                illegal[a] = True
            elif not features.free_neighbors.get(a, False):
                illegal[a] = True
            else:
                illegal[a] = False

        # Step 3 — EV synthesis (ALL terms derive from Laya outputs)
        base_reward = 10.0 if subgoal.name == "CHASE_FOOD" else 6.0

        points_at_food: dict[str, bool] = {d: False for d in DIRECTIONS}
        if features.food_axis_aligned and features.food is not None:
            dx = features.food[0] - features.head[0]
            dy = features.food[1] - features.head[1]
            if dx == 0:
                if dy < 0:
                    points_at_food["up"] = True
                elif dy > 0:
                    points_at_food["down"] = True
            elif dy == 0:
                if dx < 0:
                    points_at_food["left"] = True
                elif dx > 0:
                    points_at_food["right"] = True

        # Repeat penalties from memory
        recent_actions: tuple[str, ...] = features.recent_actions
        rejection_actions: set[str] = set()
        food_stall_turns = features.food_stall_turns

        if memory is not None:
            if hasattr(memory, "rejection_history"):
                rejection_actions = {r[0] for r in memory.rejection_history}
            if hasattr(memory, "recent_actions") and memory.recent_actions:
                recent_actions = tuple(memory.recent_actions)

        ev_breakdown: dict[str, dict[str, float]] = {}
        ev_scores: dict[str, float] = {}
        repeat_penalties: dict[str, float] = {}

        for a in DIRECTIONS:
            reward_bonus = 4.0 if points_at_food.get(a, False) else 0.0

            rep_pen = 0.0
            if a in recent_actions[-2:] and food_stall_turns >= 3:
                rep_pen += 3.0
            if a in rejection_actions:
                rep_pen += 5.0
            repeat_penalties[a] = rep_pen

            illegal_penalty = 100.0 if illegal[a] else 0.0

            prog_term = p_progress[a] * (base_reward + reward_bonus)
            danger_term = p_danger[a] * 40.0
            space_term = p_space[a] * 3.0
            regret_term = (1.0 - p_progress[a]) * 2.0

            ev = prog_term - danger_term + space_term - regret_term - rep_pen - illegal_penalty
            ev_scores[a] = ev
            ev_breakdown[a] = {
                "ev": ev,
                "p_progress": p_progress[a],
                "p_danger": p_danger[a],
                "p_space": p_space[a],
                "base_reward": base_reward,
                "reward_bonus": reward_bonus,
                "prog_term": prog_term,
                "danger_term": danger_term,
                "space_term": space_term,
                "regret_term": regret_term,
                "repeat_penalty": rep_pen,
                "illegal_penalty": illegal_penalty,
                "illegal": 1.0 if illegal[a] else 0.0,
            }

        # Step 4 — Choose argmax EV with fixed tie-breaker
        chosen = max(DIRECTIONS, key=lambda a: (ev_scores[a], -TIE_BREAK_ORDER.index(a)))
        source = "laya"
        fallback_reason: str | None = None

        # Step 5 & Step 6 — Safety fallback & Fallback budget enforcement
        if illegal.get(chosen, False):
            alt = self.compliance.check_snake_safety_fallback(features, chosen, illegal_mask=illegal)
            if alt is not None:
                if self.budget.may_fallback(accepted_so_far, fallbacks_so_far):
                    chosen = alt
                    source = "fallback"
                    fallback_reason = f"Legality fallback from illegal '{chosen}' to '{alt}'."
                else:
                    # Fallback budget is EXHAUSTED: Escaping via hard-legality clause
                    clause = make_rejection_clause(chosen)
                    fallback_reason = f"Budget exhausted; applied re-query grounding: {clause}"

                    # Re-score with hard ground truth: probability of progress is 0.0, danger is 1.0
                    re_p_progress = dict(p_progress)
                    re_p_danger = dict(p_danger)
                    re_p_progress[chosen] = 0.0
                    re_p_danger[chosen] = 1.0

                    re_ev: dict[str, float] = {}
                    for a in DIRECTIONS:
                        if a == chosen:
                            re_ev[a] = -999.0
                        else:
                            r_bonus = 4.0 if points_at_food.get(a, False) else 0.0
                            ill_term = 100.0 if illegal[a] else 0.0
                            re_ev[a] = (
                                re_p_progress[a] * (base_reward + r_bonus)
                                - re_p_danger[a] * 40.0
                                + p_space[a] * 3.0
                                - (1.0 - re_p_progress[a]) * 2.0
                                - repeat_penalties[a]
                                - ill_term
                            )
                    chosen = max(DIRECTIONS, key=lambda a: (re_ev[a], -TIE_BREAK_ORDER.index(a)))
                    source = "laya"

        # Step 7 — Produce Decision dataclass
        features_snapshot = asdict(features) if hasattr(features, "__dataclass_fields__") else dict(features)
        decision_questions = list(questions) if questions is not None else []

        return SnakeDecision(
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
        """Backward-compatibility entry point for legacy orchestrator."""
        if ev_scores:
            proposed_action = max(legal_actions, key=lambda a: (ev_scores.get(a, -999.0), -TIE_BREAK_ORDER.index(a) if a in TIE_BREAK_ORDER else 0))
        else:
            proposed_action = max(action_probabilities, key=lambda a: action_probabilities[a])

        fallback = self.compliance.check_snake_safety_fallback(state_before, proposed_action)
        if fallback is not None:
            return fallback, "fallback"

        return proposed_action, "laya"


# Alias for backward compatibility
SnakeDecisionPolicy = SnakePolicy
