"""Timing Utilities — Genesis 2.0 Track EVA 02.

Strict monotonic clock timing per PDF §6:
- decide_ms: from receiving game state to Laya returning probability distribution.
- api_ms: from sending move POST to game response arriving.
- turn_ms: from previous move's response arriving to this move's response arriving.
           For the first move, measured from the start-game response arriving.
All values are reported as integers in milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TimingSnapshot:
    """Captured timing breakdown for a single move turn."""
    decide_ms: int
    api_ms: int
    turn_ms: int

    def to_dict(self) -> dict[str, int]:
        return {
            "decide_ms": int(self.decide_ms),
            "api_ms": int(self.api_ms),
            "turn_ms": int(self.turn_ms),
        }


class MonotonicTurnTimer:
    """Manages turn lifecycle timing using time.perf_counter()."""

    def __init__(self, initial_turn_start: float | None = None) -> None:
        # Turn start anchor: previous turn's response arrival
        self._prev_turn_response_time: float = (
            initial_turn_start if initial_turn_start is not None else time.perf_counter()
        )
        self._decide_start: float | None = None
        self._decide_ms: int = 0
        self._api_start: float | None = None
        self._api_ms: int = 0

    def start_decision(self) -> None:
        """Call immediately when game state is received to begin decision timing."""
        self._decide_start = time.perf_counter()

    def end_decision(self) -> int:
        """Call immediately after Laya returns probability answer."""
        if self._decide_start is None:
            self._decide_ms = 0
        else:
            self._decide_ms = int(round((time.perf_counter() - self._decide_start) * 1000.0))
        return self._decide_ms

    def start_api(self) -> None:
        """Call immediately before dispatching the POST move HTTP request."""
        self._api_start = time.perf_counter()

    def end_api(self, api_ms_override: float | None = None) -> int:
        """Call immediately after POST response is received."""
        if api_ms_override is not None:
            self._api_ms = int(round(api_ms_override))
        elif self._api_start is not None:
            self._api_ms = int(round((time.perf_counter() - self._api_start) * 1000.0))
        else:
            self._api_ms = 0
        return self._api_ms

    def complete_turn(self, api_ms: float | None = None) -> TimingSnapshot:
        """Conclude turn, compute turn_ms from prev turn response, and advance anchor."""
        now = time.perf_counter()
        if api_ms is not None:
            self._api_ms = int(round(api_ms))
        turn_ms = int(round((now - self._prev_turn_response_time) * 1000.0))
        self._prev_turn_response_time = now

        snapshot = TimingSnapshot(
            decide_ms=self._decide_ms,
            api_ms=self._api_ms,
            turn_ms=turn_ms,
        )
        return snapshot
