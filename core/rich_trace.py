"""Rich Trace Logger — Genesis 2.0 Track EVA 02.

Stores comprehensive internal telemetry in rich_trace_<session_id>.jsonl alongside
the official minimal log. Captures:
- Raw Laya outputs
- All 4 typed hypothesis probabilities per action
- Anchor probe calibration results
- Multi-objective EV breakdown
- Selected tactical subgoal and feature vector
- Memory deltas & rejection tracking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.official_logger import canonical_json, utc_now_iso


class RichTraceLogger:
    """Writes internal diagnostic audit records to rich_trace_<session_id>.jsonl."""

    def __init__(self, session_dir: Path, session_id: str) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = session_id
        self._trace_path = self._session_dir / f"rich_trace_{session_id}.jsonl"
        self._file_handle: Any = None

    def open(self) -> Path:
        """Open trace file for append."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()
        self._file_handle = open(self._trace_path, "a", encoding="utf-8")
        return self._trace_path.resolve()

    def log_trace(
        self,
        *,
        step: int,
        action: str,
        result: str,
        raw_laya_output: Any,
        typed_probabilities: dict[str, dict[str, float]],
        anchor_probes: list[dict[str, Any]],
        ev_breakdown: dict[str, Any],
        subgoal: Any,
        features: dict[str, Any],
        memory_delta: dict[str, Any] | None = None,
        calibration_drift: float | None = None,
    ) -> None:
        """Append one structured diagnostic trace record."""
        if self._file_handle is None or self._file_handle.closed:
            self.open()

        record = {
            "session_id": self._session_id,
            "step": step,
            "ts": utc_now_iso(),
            "action": action,
            "result": result,
            "raw_laya_output": raw_laya_output,
            "typed_probabilities": typed_probabilities,
            "anchor_probes": anchor_probes,
            "ev_breakdown": ev_breakdown,
            "subgoal": str(subgoal),
            "features": features,
            "memory_delta": memory_delta or {},
            "calibration_drift": calibration_drift,
        }

        self._file_handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._file_handle.flush()

    def close(self) -> None:
        """Flush and close trace file."""
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()
