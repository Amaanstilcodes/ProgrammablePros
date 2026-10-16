"""Snake Hypothesis Builder — Genesis 2.0 Track EVA 02.

Constructs 4 typed binary hypothesis queries evaluated by Laya:
1. is_legal: Action will not collide with wall or self
2. toward_subgoal: Action reduces Manhattan distance to food
3. is_dangerous: Action enters a constrained corridor or dead end
4. is_novel: Action enters an unvisited cell
"""

from __future__ import annotations

from typing import Any

from games.snake.subgoals import SnakeSubgoal


class SnakeHypothesisBuilder:
    """Formulates typed natural language hypothesis queries for Laya evaluation."""

    def build_hypotheses(
        self,
        features: dict[str, Any],
        subgoal: SnakeSubgoal,
        legal_actions: list[str],
    ) -> dict[str, Any]:
        player_pos = features.get("player_pos", (0, 0))
        facing = features.get("facing", "right")
        bearing = features.get("bearing", "HERE")
        dist = features.get("distance_to_goal", 0)
        local = features.get("egocentric_local", {})

        context = (
            f"POV: snake head at ({player_pos[0]},{player_pos[1]}), facing {facing}. "
            f"LOCAL: AHEAD: {local.get('AHEAD')}, LEFT: {local.get('LEFT')}, "
            f"RIGHT: {local.get('RIGHT')}, BEHIND: {local.get('BEHIND')}. "
            f"OBJECTIVE: FOOD(bearing={bearing}, dist={dist})."
        )

        queries_by_action: dict[str, dict[str, str]] = {}
        flat_questions: list[str] = []

        for act in legal_actions:
            act_clean = act.strip().lower()

            q1 = (
                f"Context: {context} "
                f"Hypothesis: Executing action '{act_clean}' will result in a legal move to a free cell. "
                f"Calibrated probability?"
            )
            q2 = (
                f"Context: {context} "
                f"Hypothesis: Executing action '{act_clean}' will reduce distance to {subgoal.target_type}. "
                f"Calibrated probability?"
            )
            q3 = (
                f"Context: {context} "
                f"Hypothesis: Executing action '{act_clean}' will cause a collision or enter a trap. "
                f"Calibrated probability?"
            )
            q4 = (
                f"Context: {context} "
                f"Hypothesis: Executing action '{act_clean}' will move into an unvisited cell. "
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
