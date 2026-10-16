"""Step Ledger Module — Genesis 2.0 Track EVA 02.

Manages step numbering, move acceptance/rejection accounting, and caps.
Supports the two official interpretations of rejected-move step numbering:

Policy A (DEFAULT): "repeat_last_applied"
  - next_step starts at 1
  - on APPLIED: log step = next_step; next_step += 1
  - on REJECTED: log step = max(1, next_step - 1); do NOT advance
  - Result: applied steps form an unbroken 1..N sequence without gaps.

Policy B (ALTERNATE): "pending_ordinal"
  - on REJECTED: log step = next_step; do NOT advance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RejectedStepPolicy = Literal["repeat_last_applied", "pending_ordinal"]


@dataclass
class StepLedger:
    """Tracks applied and rejected move sequences, emitting compliant step indices."""

    policy: RejectedStepPolicy = "repeat_last_applied"
    max_accepted_steps: int = 300

    def __post_init__(self) -> None:
        self._next_step: int = 1
        self._accepted_count: int = 0
        self._rejected_count: int = 0
        self._last_applied_step: int = 0

    @property
    def accepted_count(self) -> int:
        """Total number of applied (accepted) moves."""
        return self._accepted_count

    @property
    def rejected_count(self) -> int:
        """Total number of rejected moves."""
        return self._rejected_count

    @property
    def total_attempts(self) -> int:
        """Total moves dispatched to server."""
        return self._accepted_count + self._rejected_count

    @property
    def is_cap_reached(self) -> bool:
        """True when accepted moves reach the mandated cap."""
        return self._accepted_count >= self.max_accepted_steps

    def record_move(self, result: str) -> int:
        """Process move result ('applied' or 'rejected') and return compliant step number.

        Args:
            result: 'applied' or 'rejected'.

        Returns:
            The integer step value to write in the JSONL log line.
        """
        res_clean = result.strip().lower()

        if res_clean == "applied":
            step_to_log = self._next_step
            self._last_applied_step = step_to_log
            self._next_step += 1
            self._accepted_count += 1
            return step_to_log

        # Rejected move
        self._rejected_count += 1
        if self.policy == "repeat_last_applied":
            # Policy A: repeat previous applied step, or 1 if rejected before any applied move
            return max(1, self._last_applied_step)
        elif self.policy == "pending_ordinal":
            # Policy B: use pending next_step without incrementing
            return self._next_step
        else:
            raise ValueError(f"Unknown RejectedStepPolicy: {self.policy}")
