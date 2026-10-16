"""Advanced Expected Value (EV) Calculator — Genesis 2.0 Track EVA 02.

Computes multi-objective, risk-adjusted Expected Value scores across 4 calibrated
dimensions:
1. Legality: Filters impossible moves into walls / invalid transitions
2. Subgoal Progress: Positive utility for reducing metric distance to active objective
3. Danger Avoidance: Asymmetric penalties for hazards, collisions, and damage
4. Novelty Exploration: Information gain bonus for visiting unmapped cells
5. Repeat Failure Suppression: History penalty for recently failed actions at current position
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class AdvancedEVCalculator:
    """Multi-objective Expected Value calculator for autonomous decision making.

    Formula:
        EV(a) = P(legal) * [
            (P(subgoal) * W_subgoal - (1 - P(subgoal)) * W_step)
            - P(danger) * W_danger
            + P(novel) * W_novel
        ] - (1 - P(legal)) * W_illegal - repeat_penalty
    """

    def __init__(
        self,
        w_subgoal: float = 12.0,
        w_step_fail: float = 2.0,
        w_danger: float = 30.0,
        w_novel: float = 3.0,
        w_illegal: float = 25.0,
        w_repeat: float = 15.0,
    ) -> None:
        self.w_subgoal: float = w_subgoal
        self.w_step_fail: float = w_step_fail
        self.w_danger: float = w_danger
        self.w_novel: float = w_novel
        self.w_illegal: float = w_illegal
        self.w_repeat: float = w_repeat

    @staticmethod
    def _normalize_pos(pos: Any) -> Tuple[int, int]:
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
        if isinstance(pos, (int, float)):
            return int(pos), 0
        return 0, 0

    def calculate(
        self,
        current_pos: Any,
        legal_actions: List[str],
        probs: Dict[str, Any],
        recent_failures: Optional[List[Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Calculate multi-objective Expected Value for all candidate actions.

        Args:
            current_pos: Agent (x, y) coordinates or scalar position.
            legal_actions: List of valid candidate action strings.
            probs: Nested mapping of action -> 4 typed probabilities or single float:
                   e.g., {"up": {"is_legal": 0.95, "toward_subgoal": 0.8, "is_dangerous": 0.05, "is_novel": 0.6}}
            recent_failures: List of failure records or action strings at current position.

        Returns:
            Tuple of (chosen_action, ev_breakdown_dict).
        """
        curr_xy = self._normalize_pos(current_pos)
        ev_breakdown: Dict[str, Any] = {}

        failed_action_counts: Dict[str, int] = {}
        if recent_failures:
            for entry in recent_failures:
                if isinstance(entry, dict):
                    act = str(entry.get("action", "")).strip().lower()
                    entry_pos = self._normalize_pos(entry.get("pos", curr_xy))
                    if entry_pos == curr_xy and act:
                        failed_action_counts[act] = failed_action_counts.get(act, 0) + 1
                elif isinstance(entry, str):
                    act = entry.strip().lower()
                    failed_action_counts[act] = failed_action_counts.get(act, 0) + 1

        for act in legal_actions:
            act_clean = act.strip().lower()
            act_data = probs.get(act_clean, probs.get(act, {}))

            if isinstance(act_data, dict):
                p_legal = float(act_data.get("is_legal", 1.0))
                p_subgoal = float(act_data.get("toward_subgoal", 0.5))
                p_danger = float(act_data.get("is_dangerous", 0.0))
                p_novel = float(act_data.get("is_novel", 0.0))
            elif isinstance(act_data, (int, float)):
                p_legal = 1.0
                p_subgoal = float(act_data)
                p_danger = 0.0
                p_novel = 0.0
            else:
                p_legal, p_subgoal, p_danger, p_novel = 1.0, 0.5, 0.0, 0.0

            subgoal_ev = (p_subgoal * self.w_subgoal) - ((1.0 - p_subgoal) * self.w_step_fail)
            danger_penalty = p_danger * self.w_danger
            novelty_bonus = p_novel * self.w_novel
            illegal_penalty = (1.0 - p_legal) * self.w_illegal

            repeat_count = failed_action_counts.get(act_clean, 0)
            repeat_penalty = repeat_count * self.w_repeat

            total_ev = (
                (p_legal * (subgoal_ev - danger_penalty + novelty_bonus))
                - illegal_penalty
                - repeat_penalty
            )

            ev_breakdown[act_clean] = {
                "ev": round(total_ev, 4),
                "p_legal": round(p_legal, 4),
                "p_subgoal": round(p_subgoal, 4),
                "p_danger": round(p_danger, 4),
                "p_novel": round(p_novel, 4),
                "subgoal_utility": round(subgoal_ev, 4),
                "danger_penalty": round(danger_penalty, 4),
                "novelty_bonus": round(novelty_bonus, 4),
                "illegal_penalty": round(illegal_penalty, 4),
                "repeat_penalty": round(repeat_penalty, 4),
            }

        if not ev_breakdown:
            best_action = legal_actions[0] if legal_actions else "up"
        else:
            best_action = max(ev_breakdown, key=lambda a: ev_breakdown[a]["ev"])

        return best_action, ev_breakdown
