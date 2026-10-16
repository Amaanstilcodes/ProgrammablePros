"""2048 Hypothesis Builder — Genesis 2.0 Track EVA 02.

Constructs 4 typed binary hypothesis queries evaluated by Laya for 2048:
1. is_legal: Action will alter the board state (not be rejected)
2. toward_subgoal: Action will successfully combine tiles or protect corner anchor
3. is_dangerous: Action will scatter tiles or displace the highest tile from corner
4. is_novel: Action will create a new maximum tile or clear fresh space
"""

from __future__ import annotations

from typing import Any

from games.game2048.subgoals import Game2048Subgoal


class Game2048HypothesisBuilder:
    """Formulates typed natural language hypothesis queries for Laya evaluation."""

    def build_hypotheses(
        self,
        features: dict[str, Any],
        subgoal: Game2048Subgoal,
        legal_actions: list[str],
    ) -> dict[str, Any]:
        empty = features.get("empty_cells", 0)
        max_t = features.get("max_tile", 2)
        corner = features.get("is_corner", False)

        context = (
            f"BOARD: 4x4 grid with max_tile={max_t} (in_corner={corner}), "
            f"empty_cells={empty}/16, merges_available={features.get('available_merges', 0)}. "
            f"SUBGOAL: {subgoal.description}."
        )

        queries_by_action: dict[str, dict[str, str]] = {}
        flat_questions: list[str] = []

        for act in legal_actions:
            act_clean = act.strip().lower()

            q1 = (
                f"Context: {context} "
                f"Hypothesis: Swiping '{act_clean}' will produce a valid board change without rejection. "
                f"Calibrated probability?"
            )
            q2 = (
                f"Context: {context} "
                f"Hypothesis: Swiping '{act_clean}' will achieve subgoal {subgoal.target_type}. "
                f"Calibrated probability?"
            )
            q3 = (
                f"Context: {context} "
                f"Hypothesis: Swiping '{act_clean}' will compromise the anchor corner or lock the grid. "
                f"Calibrated probability?"
            )
            q4 = (
                f"Context: {context} "
                f"Hypothesis: Swiping '{act_clean}' will create a higher value tile. "
                f"Calibrated probability?"
            )

            queries_by_action[act_clean] = {
                "is_legal": q1,
                "toward_subgoal": q2,
                "is_dangerous": q3,
                "is_novel": q4,
            }
            flat_questions.extend([q1, q2, q3, q4])

        return {
            "context": context,
            "queries_by_action": queries_by_action,
            "flat_questions": flat_questions,
            "primary_questions": [queries_by_action[a]["toward_subgoal"] for a in legal_actions],
        }
