"""Snake Game Memory — Genesis 2.0 Track EVA 02 / Phase 2a.

Maintains spatial occupancy, visit counts, growth events, rejection history,
and provides failure-adaptive context warnings.
"""

from __future__ import annotations

from collections import deque
from typing import Any

OPPOSITES: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class SnakeMemory:
    """Stateful memory for tracking board visits, food stalls, and rejections."""

    def __init__(self) -> None:
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.body_history: deque[tuple[int, int]] = deque(maxlen=8)
        self.recent_outcomes: deque[tuple[str, tuple[int, int] | None, tuple[int, int] | None, str]] = deque(maxlen=3)
        self.food_stall_turns: int = 0
        self.rejection_history: deque[tuple[str, str]] = deque(maxlen=5)
        self.last_known_food: tuple[int, int] | None = None
        self.inferred_growth_events: list[tuple[int, tuple[int, int]]] = []
        
        self._step: int = 0
        self._last_reduced_space: bool = False
        self._last_action: str = ""

    @property
    def visits(self) -> dict[tuple[int, int], int]:
        """Backward-compatibility alias."""
        return self.visit_counts

    def get_visit_count(self, pos: tuple[int, int]) -> int:
        """Return visit count for cell."""
        return self.visit_counts.get(pos, 0)

    def _extract_head(self, state: Any) -> tuple[int, int] | None:
        if state is None:
            return None
        head = getattr(state, "head", None)
        if head is not None and isinstance(head, (list, tuple)) and len(head) >= 2:
            return (int(head[0]), int(head[1]))
        if isinstance(state, dict):
            h = state.get("head") or (state.get("snake")[0] if state.get("snake") else None)
            if h is not None and len(h) >= 2:
                return (int(h[0]), int(h[1]))
        return None

    def _extract_body_length(self, state: Any) -> int:
        if state is None:
            return 0
        body = getattr(state, "body", None)
        if body is not None:
            return len(body)
        if isinstance(state, dict):
            s = state.get("snake") or state.get("body")
            if s:
                return len(s)
        return 1

    def _extract_food(self, state: Any) -> tuple[int, int] | None:
        if state is None:
            return None
        f = getattr(state, "food", None)
        if f is not None and isinstance(f, (list, tuple)) and len(f) >= 2:
            return (int(f[0]), int(f[1]))
        if isinstance(state, dict):
            raw = state.get("food")
            if raw is not None and isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return (int(raw[0]), int(raw[1]))
        return None

    def _count_free(self, state: Any) -> int:
        head = self._extract_head(state)
        if head is None:
            return 4
        direction = getattr(state, "direction", None) or (state.get("direction") if isinstance(state, dict) else "right")
        reversal = OPPOSITES.get(str(direction).lower(), "")
        
        body_raw = getattr(state, "body", None) or (state.get("snake") if isinstance(state, dict) else None) or []
        body_set = {(int(p[0]), int(p[1])) for p in body_raw}
        
        w = getattr(state, "width", None) or (state.get("width") if isinstance(state, dict) else None) or 16
        h = getattr(state, "height", None) or (state.get("height") if isinstance(state, dict) else None) or 16

        free = 0
        for d, delta in DELTAS.items():
            if d == reversal:
                continue
            nx, ny = head[0] + delta[0], head[1] + delta[1]
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in body_set:
                free += 1
        return free

    def observe_transition(
        self,
        before: Any,
        action: str,
        after: Any,
        result: str,
    ) -> None:
        """Observe step transition and update tracking invariants."""
        act_clean = action.strip().lower()
        self._last_action = act_clean
        head_before = self._extract_head(before)
        head_after = self._extract_head(after)

        # Track food if exposed
        food_before = self._extract_food(before)
        food_after = self._extract_food(after)
        if food_after is not None:
            self.last_known_food = food_after
        elif food_before is not None:
            self.last_known_food = food_before

        if result == "rejected":
            # Rejection: append to rejection history and do NOT update visit counts
            before_dir = getattr(before, "direction", None) or (before.get("direction") if isinstance(before, dict) else None)
            reason = "reversal" if (before_dir and act_clean == OPPOSITES.get(str(before_dir).lower())) else "rejected_by_game"
            self.rejection_history.append((act_clean, reason))
            self.recent_outcomes.append((act_clean, head_before, head_after or head_before, "rejected"))
            self._last_reduced_space = False
            self._step += 1
            return

        # Move accepted: determine body growth
        len_before = self._extract_body_length(before)
        len_after = self._extract_body_length(after)

        if len_after > len_before:
            # Growth event at new head position
            if head_after is not None:
                self.inferred_growth_events.append((self._step, head_after))
            self.food_stall_turns = 0
        else:
            self.food_stall_turns += 1

        # Update visit counts for the new head
        if head_after is not None:
            self.visit_counts[head_after] = self.visit_counts.get(head_after, 0) + 1
            self.body_history.append(head_after)

        # Space reduction metric
        free_before = self._count_free(before)
        free_after = self._count_free(after)
        self._last_reduced_space = (free_after < free_before)

        # Append outcome tuple
        self.recent_outcomes.append((act_clean, head_before, head_after, "applied"))
        self._step += 1

    def observe(self, before: Any, action: str, after: Any) -> None:
        """Backward-compatibility wrapper for orchestrator."""
        self.observe_transition(before, action, after, "applied")

    def warning_text(self) -> str:
        """Return at most ONE failure-adaptive warning string, in strict priority order:

        1. Last action rejected as reversal.
        2. Last action rejected for other reasons.
        3. No food eaten for >= 5 turns.
        4. Last move reduced free_neighbors.
        5. Otherwise empty string.
        """
        # Check if the immediate last outcome was rejected
        if self.recent_outcomes and self.recent_outcomes[-1][3] == "rejected":
            last_act, _, _, _ = self.recent_outcomes[-1]
            # Check rejection history for reason
            last_reason = self.rejection_history[-1][1] if self.rejection_history else ""
            if last_reason == "reversal":
                return f"WARNING: '{last_act}' reverses the current direction and is illegal."
            return f"WARNING: '{last_act}' was rejected by the game. Do not repeat it."

        # Priority 3: Food stall >= 5 turns
        if self.food_stall_turns >= 5:
            return (
                f"WARNING: No food eaten for {self.food_stall_turns} turns. "
                f"Prefer safer positioning over greedy approach."
            )

        # Priority 4: Last move reduced free_neighbors
        if self._last_reduced_space and self._last_action:
            return f"WARNING: '{self._last_action}' reduced open space. Prefer preserving escape room."

        return ""

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable snapshot for rich trace logging."""
        return {
            "visit_counts": {f"{k[0]},{k[1]}": v for k, v in self.visit_counts.items()},
            "body_history": list(self.body_history),
            "recent_outcomes": list(self.recent_outcomes),
            "food_stall_turns": self.food_stall_turns,
            "rejection_history": list(self.rejection_history),
            "last_known_food": self.last_known_food,
            "inferred_growth_events": list(self.inferred_growth_events),
            "step": self._step,
        }
