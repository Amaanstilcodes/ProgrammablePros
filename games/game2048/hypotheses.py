"""2048 Hypothesis Builder — Genesis 2.0 Track EVA 02 / Phase 2b.

Constructs typed natural-language hypothesis queries evaluated by Laya:
- progress (PRIMARY — logged probability vector)
- danger (congestion / game over risk)
- structure (corner anchor and snake chain preservation)
- Anchor probes (2 drift monitoring questions appended for rich trace)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from games.game2048.features import (
    Board2048Features,
    Game2048FeatureExtractor,
    format_context_string,
    is_board_changing,
)
from games.game2048.subgoals import Subgoal2048

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")


@dataclass(frozen=True)
class HypothesisMap2048:
    """Deterministic bidirectional mapping between (action, kind) and question index."""

    mapping: dict[tuple[str, str], int]
    actions: tuple[str, ...] = DIRECTIONS
    kinds: tuple[str, ...] = ("progress", "danger", "structure")

    def get_index(self, action: str, kind: str) -> int:
        return self.mapping[(action.lower(), kind.lower())]

    def __getitem__(self, key: tuple[str, str]) -> int:
        return self.get_index(key[0], key[1])


@dataclass
class HypothesisBatch2048:
    """Container holding synthesized hypothesis questions and metadata."""

    context: str
    decision_questions: list[str]
    anchor_questions: list[str]
    anchor_expectations: list[float]
    all_questions: list[str]
    hypothesis_map: HypothesisMap2048
    subgoal: Subgoal2048

    def __getitem__(self, key: str) -> Any:
        if key == "context":
            return self.context
        if key == "flat_questions":
            return self.decision_questions
        if key == "primary_questions":
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
                    "is_legal": f"Hypothesis: sliding '{a}' changes the board.",
                }
                if "structure" in self.hypothesis_map.kinds and (a, "structure") in self.hypothesis_map.mapping:
                    queries[a]["is_novel"] = self.decision_questions[self.hypothesis_map.get_index(a, "structure")]
            return queries
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class Board2048HypothesisBuilder:
    """Synthesizes deterministic 8-question (lean) or 12-question (full) batches plus anchors."""

    def __init__(self, hypothesis_mode: str = "full") -> None:
        if hypothesis_mode not in ("lean", "full"):
            raise ValueError(f"Invalid hypothesis_mode '{hypothesis_mode}'. Must be 'lean' or 'full'.")
        self.hypothesis_mode = hypothesis_mode

    def build(
        self,
        features: Board2048Features,
        subgoal: Subgoal2048,
        memory: Any = None,
    ) -> HypothesisBatch2048:
        """Build deterministic hypothesis set."""
        context = format_context_string(features, subgoal.name)

        kinds = ("progress", "danger") if self.hypothesis_mode == "lean" else ("progress", "danger", "structure")

        decision_questions: list[str] = []
        mapping: dict[tuple[str, str], int] = {}

        for act in DIRECTIONS:
            q_progress = (
                f"Context: {context}. "
                f"Hypothesis: Sliding '{act}' is legal (it changes the board) AND advances the objective: {subgoal.goal_phrase}. "
                f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
            )
            mapping[(act, "progress")] = len(decision_questions)
            decision_questions.append(q_progress)

            q_danger = (
                f"Context: {context}. "
                f"Hypothesis: Sliding '{act}' leaves the board with zero or one empty cell AND no immediately available merge, risking game over. "
                f"What is the calibrated probability (0.0 to 1.0) that this is true?"
            )
            mapping[(act, "danger")] = len(decision_questions)
            decision_questions.append(q_danger)

            if self.hypothesis_mode == "full":
                q_structure = (
                    f"Context: {context}. "
                    f"Hypothesis: After sliding '{act}', the largest tile remains in the bottom-left corner and the snake-chain monotonicity is maintained or improved. "
                    f"What is the calibrated probability (0.0 to 1.0) that this is true?"
                )
                mapping[(act, "structure")] = len(decision_questions)
                decision_questions.append(q_structure)

        # ANCHOR PROBES (appended for rich trace only, never alters decision vector)
        # Anchor A: direction that does not change board (or impossible action)
        no_op_dir: str | None = None
        for a in DIRECTIONS:
            if not is_board_changing(features.board, a):
                no_op_dir = a
                break

        if no_op_dir is not None:
            anchor_a = f"Hypothesis: sliding '{no_op_dir}' changes the board."
        else:
            anchor_a = f"Hypothesis: sliding 'none' changes the board."
        expected_a = 0.0

        # Anchor B: tautology check
        anchor_b = f"Hypothesis: the largest tile on this board is {features.max_tile}."
        expected_b = 1.0

        anchor_questions = [anchor_a, anchor_b]
        anchor_expectations = [expected_a, expected_b]
        all_questions = list(decision_questions) + anchor_questions

        hypo_map = HypothesisMap2048(
            mapping=mapping,
            actions=DIRECTIONS,
            kinds=kinds,
        )

        return HypothesisBatch2048(
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
    ) -> HypothesisBatch2048:
        """Backward-compatibility wrapper for orchestrator."""
        if not isinstance(features_or_dict, Board2048Features):
            extractor = Game2048FeatureExtractor()
            features = extractor.extract_features(features_or_dict, None)
        else:
            features = features_or_dict

        if not isinstance(subgoal, Subgoal2048):
            subgoal = Subgoal2048(
                name=getattr(subgoal, "target_type", "KEEP_CORNER"),
                goal_phrase=getattr(subgoal, "description", "keeping the largest tile fixed in the bottom-left corner"),
                priority=3,
                rationale="Fallback subgoal adaptation.",
            )

        return self.build(features, subgoal, None)


# Alias for backward compatibility
Game2048HypothesisBuilder = Board2048HypothesisBuilder
