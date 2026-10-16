"""Snake Hypothesis Builder — Genesis 2.0 Track EVA 02 / Phase 2a.

Constructs typed natural-language hypothesis queries for Laya probability evaluation:
- progress (PRIMARY — logged probability vector)
- danger (immediate wall or body collision)
- space (at least two legal directions on next turn)
- Anchor probes (2 drift monitoring questions appended for rich trace)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from games.snake.features import FeatureExtractor, SnakeFeatures, format_context_string
from games.snake.subgoals import Subgoal

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

OPPOSITES: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass(frozen=True)
class HypothesisMap:
    """Deterministic bidirectional mapping between (action, kind) and question index."""

    mapping: dict[tuple[str, str], int]
    actions: tuple[str, ...] = DIRECTIONS
    kinds: tuple[str, ...] = ("progress", "danger", "space")

    def get_index(self, action: str, kind: str) -> int:
        return self.mapping[(action.lower(), kind.lower())]

    def __getitem__(self, key: tuple[str, str]) -> int:
        return self.get_index(key[0], key[1])


@dataclass
class HypothesisBatch:
    """Container holding synthesized hypothesis questions and metadata."""

    context: str
    decision_questions: list[str]
    anchor_questions: list[str]
    anchor_expectations: list[float]
    all_questions: list[str]
    hypothesis_map: HypothesisMap
    subgoal: Subgoal

    # Backward-compatibility dictionary access
    def __getitem__(self, key: str) -> Any:
        if key == "context":
            return self.context
        if key == "flat_questions":
            return self.decision_questions
        if key == "primary_questions":
            # Primary questions are progress questions
            return [
                self.decision_questions[self.hypothesis_map.get_index(a, "progress")]
                for a in DIRECTIONS
            ]
        if key == "queries_by_action":
            queries: dict[str, dict[str, str]] = {}
            for a in DIRECTIONS:
                queries[a] = {
                    "toward_subgoal": self.decision_questions[self.hypothesis_map.get_index(a, "progress")],
                    "is_dangerous": self.decision_questions[self.hypothesis_map.get_index(a, "danger")],
                    "is_legal": f"Hypothesis: moving '{a}' is legal.",
                }
                if ("space" in self.hypothesis_map.kinds and
                        (a, "space") in self.hypothesis_map.mapping):
                    queries[a]["is_novel"] = self.decision_questions[self.hypothesis_map.get_index(a, "space")]
            return queries
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class DriftMonitor:
    """Monitors anchor probe outputs for calibration drift (>0.25 error for 3 turns)."""

    def __init__(self, error_threshold: float = 0.25, consecutive_limit: int = 3) -> None:
        self.error_threshold = error_threshold
        self.consecutive_limit = consecutive_limit
        self.drift_streak: int = 0

    def check(self, anchor_outputs: list[float], expectations: list[float]) -> str | None:
        """Check if drift threshold is exceeded on any anchor probe."""
        if not anchor_outputs or not expectations:
            return None

        any_drift = False
        for out, exp in zip(anchor_outputs, expectations):
            if abs(out - exp) > self.error_threshold:
                any_drift = True
                break

        if any_drift:
            self.drift_streak += 1
        else:
            self.drift_streak = 0

        if self.drift_streak >= self.consecutive_limit:
            msg = (
                f"[CALIBRATION_DRIFT] Anchor probes deviated > {self.error_threshold} "
                f"for {self.drift_streak} consecutive turns!"
            )
            print(msg, file=sys.stderr)
            return msg

        return None


class SnakeHypothesisBuilder:
    """Constructs deterministic 8-question (lean) or 12-question (full) batches plus anchors."""

    def __init__(self, hypothesis_mode: str = "full") -> None:
        if hypothesis_mode not in ("lean", "full"):
            raise ValueError(f"Invalid hypothesis_mode '{hypothesis_mode}'. Must be 'lean' or 'full'.")
        self.hypothesis_mode = hypothesis_mode
        self.drift_monitor = DriftMonitor()

    def build(
        self,
        features: SnakeFeatures,
        subgoal: Subgoal,
        memory: Any = None,
    ) -> HypothesisBatch:
        """Build deterministic hypothesis set."""
        context = format_context_string(features, subgoal.name)

        kinds = ("progress", "danger") if self.hypothesis_mode == "lean" else ("progress", "danger", "space")

        decision_questions: list[str] = []
        mapping: dict[tuple[str, str], int] = {}

        # Ordering MUST be deterministic: [up.progress, up.danger, up.space, down.progress, ...]
        for act in DIRECTIONS:
            # KIND = "progress"
            q_progress = (
                f"Context: {context}. "
                f"Hypothesis: From this state, moving '{act}' is legal AND advances the agent toward {subgoal.goal_phrase}. "
                f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
            )
            mapping[(act, "progress")] = len(decision_questions)
            decision_questions.append(q_progress)

            # KIND = "danger"
            q_danger = (
                f"Context: {context}. "
                f"Hypothesis: From this state, moving '{act}' causes an immediate collision with a wall or the snake's own body. "
                f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
            )
            mapping[(act, "danger")] = len(decision_questions)
            decision_questions.append(q_danger)

            # KIND = "space" (full mode only)
            if self.hypothesis_mode == "full":
                q_space = (
                    f"Context: {context}. "
                    f"Hypothesis: After moving '{act}', the snake will have at least two legally available directions on the following turn. "
                    f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
                )
                mapping[(act, "space")] = len(decision_questions)
                decision_questions.append(q_space)

        # ANCHOR PROBES (2 per turn, appended for rich trace monitoring, NEVER modifies decisions)
        # Anchor A: deterministically illegal direction (reversal is always illegal in snake)
        illegal_act = OPPOSITES.get(features.direction, "down")
        anchor_a = f"Hypothesis: moving '{illegal_act}' from this state is legal."
        expected_a = 0.0

        # Anchor B: tautology calibrated check
        anchor_b = f"Hypothesis: the snake currently occupies exactly {features.body_length} cells."
        expected_b = 1.0

        anchor_questions = [anchor_a, anchor_b]
        anchor_expectations = [expected_a, expected_b]
        all_questions = list(decision_questions) + anchor_questions

        hypo_map = HypothesisMap(
            mapping=mapping,
            actions=DIRECTIONS,
            kinds=kinds,
        )

        return HypothesisBatch(
            context=context,
            decision_questions=decision_questions,
            anchor_questions=anchor_questions,
            anchor_expectations=anchor_expectations,
            all_questions=all_questions,
            hypothesis_map=hypo_map,
            subgoal=subgoal,
        )

    def build_hypotheses(
        self,
        features_or_dict: Any,
        subgoal: Any,
        legal_actions: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> HypothesisBatch:

        """Backward-compatible entry point for orchestrator and benchmark tools."""
        if not isinstance(features_or_dict, SnakeFeatures):
            extractor = FeatureExtractor()
            features = extractor.extract_features(features_or_dict, None)
        else:
            features = features_or_dict

        if not isinstance(subgoal, Subgoal):
            subgoal = Subgoal(
                name=getattr(subgoal, "target_type", "CHASE_FOOD"),
                goal_phrase=getattr(subgoal, "description", "reaching the food while keeping an escape route open"),
                priority=3,
                rationale="Fallback subgoal adaptation.",
            )

        return self.build(features, subgoal, None)
