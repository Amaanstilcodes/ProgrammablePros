"""Decision Logger Module.

Provides append-only, non-blocking JSON Lines (JSONL) persistence for real-time
telemetry, auditability, and live Streamlit streaming without lock contention.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


class DecisionLogger:
    """Append-only streaming logger for agent decisions and Laya probability calibration records."""

    def __init__(self, log_filepath: str = "decision_log.jsonl") -> None:
        """Initialize logger with destination file path.

        Args:
            log_filepath: Path to the JSONL log file (default: "decision_log.jsonl").
        """
        self.log_filepath: str = log_filepath
        # Ensure parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(log_filepath))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    @staticmethod
    def _serialize_subgoal(subgoal: Any) -> Any:
        """Convert Subgoal dataclass or dict to JSON-serializable representation."""
        if is_dataclass(subgoal) and not isinstance(subgoal, type):
            return asdict(subgoal)
        if isinstance(subgoal, dict):
            return subgoal
        return str(subgoal)

    def append_decision(
        self,
        session_id: str,
        turn_number: int,
        raw_state: Dict[str, Any],
        compressed_context: str,
        subgoal: Any,
        exact_prompts_sent: List[str],
        raw_laya_probs: Dict[str, Dict[str, float]],
        calibrated_probs: Dict[str, Dict[str, float]],
        ev_breakdown: Dict[str, Any],
        chosen_action: str,
        next_state: Dict[str, Any],
        outcome: str,
    ) -> None:
        """Write a complete decision snapshot as a single JSON line and flush immediately.

        Args:
            session_id: Unique identifier for the game session.
            turn_number: Monotonic 1-indexed turn step number.
            raw_state: Full game environment state dictionary prior to move.
            compressed_context: Egocentric structured POV context string.
            subgoal: Active Subgoal instance or dictionary representation.
            exact_prompts_sent: List of natural language hypothesis prompts queried.
            raw_laya_probs: Model raw probability outputs keyed by action and query type.
            calibrated_probs: Platt-scaled calibrated probabilities.
            ev_breakdown: AdvancedEVCalculator score breakdown dictionary.
            chosen_action: Action selected by EV argmax.
            next_state: Resulting environment state dictionary after move execution.
            outcome: Transition evaluation (e.g., 'SUCCESS', 'FAILED_NO_PROGRESS', 'DAMAGE_TAKEN').
        """
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "session_id": session_id,
            "turn_number": turn_number,
            "raw_state": raw_state,
            "compressed_context": compressed_context,
            "subgoal": self._serialize_subgoal(subgoal),
            "exact_prompts_sent": exact_prompts_sent,
            "raw_laya_probs": raw_laya_probs,
            "calibrated_probs": calibrated_probs,
            "ev_breakdown": ev_breakdown,
            "chosen_action": chosen_action,
            "next_state": next_state,
            "outcome": outcome,
        }

        # Open in append mode and flush immediately to support live polling by viewer.py
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()

    def read_recent_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read recent decision records from the JSONL log file.

        Args:
            limit: Maximum number of most recent records to return (None for all).

        Returns:
            List of parsed decision dictionaries.
        """
        if not os.path.exists(self.log_filepath):
            return []

        records: List[Dict[str, Any]] = []
        try:
            with open(self.log_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []

        return records[-limit:] if limit is not None else records

    def clear(self) -> None:
        """Clear existing log file."""
        if os.path.exists(self.log_filepath):
            with open(self.log_filepath, "w", encoding="utf-8") as f:
                f.write("")


if __name__ == "__main__":
    logger = DecisionLogger(log_filepath="test_decision_log.jsonl")
    logger.clear()
    logger.append_decision(
        session_id="test-session-123",
        turn_number=1,
        raw_state={"player_pos": (0, 0)},
        compressed_context="POV: agent at (0,0), facing RIGHT",
        subgoal={"target_type": "KEY", "coordinates": (2, 2)},
        exact_prompts_sent=["Hypothesis: Action 'right' legal?"],
        raw_laya_probs={"right": {"is_legal": 0.88}},
        calibrated_probs={"right": {"is_legal": 0.85}},
        ev_breakdown={"right": {"ev": 5.5}},
        chosen_action="right",
        next_state={"player_pos": (1, 0)},
        outcome="SUCCESS",
    )
    logs = logger.read_recent_logs(limit=5)
    assert len(logs) == 1
    assert logs[0]["chosen_action"] == "right"
    assert logs[0]["outcome"] == "SUCCESS"
    os.remove("test_decision_log.jsonl")
    print("[DecisionLogger self-test PASSED]")
