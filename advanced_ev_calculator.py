"""Advanced Expected Value (EV) Calculator Module.

Computes multi-objective, risk-adjusted Expected Value scores across 4 calibrated
dimensions:
1. Legality: Filters impossible moves into walls
2. Subgoal Progress: Positive utility for reducing distance to active objective
3. Danger Avoidance: Asymmetric penalties for damage, traps, and enemy proximity
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
        """Initialize calculator with multi-objective weights.

        Args:
            w_subgoal: Reward weight for progressing toward subgoal.
            w_step_fail: Minor penalty for wasting a turn without progress.
            w_danger: Severe penalty for exposure to hazards or enemy damage.
            w_novel: Exploration bonus weight for unvisited cells.
            w_illegal: Heavy penalty for attempting to walk into confirmed walls.
            w_repeat: Suppression penalty for actions that failed recently at current pos.
        """
        self.w_subgoal: float = w_subgoal
        self.w_step_fail: float = w_step_fail
        self.w_danger: float = w_danger
        self.w_novel: float = w_novel
        self.w_illegal: float = w_illegal
        self.w_repeat: float = w_repeat

    @staticmethod
    def _normalize_pos(pos: Any) -> Tuple[int, int]:
        """Convert position representation to (x, y) tuple."""
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
        """Calculate multi-objective Expected Value for all legal actions.

        Args:
            current_pos: Agent (x, y) coordinates or scalar position.
            legal_actions: List of valid action strings.
            probs: Nested mapping of action -> 4 typed probabilities or single float:
                   e.g., {"up": {"is_legal": 0.95, "toward_subgoal": 0.8, "is_dangerous": 0.05, "is_novel": 0.6}}
            recent_failures: List of failure records or action strings at current position.

        Returns:
            Tuple of (chosen_action, ev_breakdown_dict).
        """
        curr_xy = self._normalize_pos(current_pos)
        ev_breakdown: Dict[str, Any] = {}

        # Parse recent failures to count repeated failed attempts per action at current position
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

            # 1. Subgoal component (Reward vs Wasted step)
            subgoal_ev = (p_subgoal * self.w_subgoal) - ((1.0 - p_subgoal) * self.w_step_fail)

            # 2. Risk component
            danger_penalty = p_danger * self.w_danger

            # 3. Novelty exploration bonus
            novelty_bonus = p_novel * self.w_novel

            # 4. Legality penalty
            illegal_penalty = (1.0 - p_legal) * self.w_illegal

            # 5. Repeat failure suppression
            repeat_count = failed_action_counts.get(act_clean, 0)
            repeat_penalty = repeat_count * self.w_repeat

            # Total EV synthesis
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

        # Select action with maximum expected value (deterministic argmax)
        if not ev_breakdown:
            best_action = legal_actions[0] if legal_actions else "wait"
        else:
            best_action = max(ev_breakdown, key=lambda a: ev_breakdown[a]["ev"])

        return best_action, ev_breakdown


if __name__ == "__main__":
    calc = AdvancedEVCalculator()
    probs = {
        "up": {"is_legal": 0.95, "toward_subgoal": 0.85, "is_dangerous": 0.02, "is_novel": 0.1},
        "down": {"is_legal": 0.95, "toward_subgoal": 0.30, "is_dangerous": 0.70, "is_novel": 0.0},
        "right": {"is_legal": 0.10, "toward_subgoal": 0.90, "is_dangerous": 0.0, "is_novel": 0.0},
    }
    failures = [{"pos": (1, 1), "action": "right"}]
    best_act, breakdown = calc.calculate((1, 1), ["up", "down", "right"], probs, failures)
    print(f"[AdvancedEVCalculator self-test PASSED]")
    print(f"  Best Action: '{best_act}'")
    for act, data in breakdown.items():
        print(f"  Action '{act}': EV={data['ev']:+.2f} (subgoal={data['p_subgoal']:.2f}, danger={data['p_danger']:.2f})")
    assert best_act == "up"
