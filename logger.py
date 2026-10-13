"""Decision Logger Module.

Records step-by-step game decisions, questions submitted to Laya, calibrated probabilities,
and chosen actions to a JSONL log file for auditing and live visualization.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional


class DecisionLogger:
    """Logs game decisions to a structured JSON Lines (.jsonl) file."""

    def __init__(self, log_path: str = "decision_log.jsonl", clear_on_init: bool = False) -> None:
        """Initialize the logger.

        Args:
            log_path: Filepath where JSONL entries will be appended.
            clear_on_init: If True, resets/empties the log file on initialization.
        """
        self.log_path: str = log_path

        if clear_on_init and os.path.exists(self.log_path):
            try:
                os.remove(self.log_path)
            except OSError as exc:
                print(f"[DecisionLogger Warning] Could not clear existing log file: {exc}")

    def log_decision(
        self,
        game_state: Dict[str, Any],
        questions_asked: Any,
        probabilities_returned: Dict[str, float],
        chosen_action: str,
        compressed_state: Optional[str] = None,
        expected_values: Optional[Dict[str, float]] = None,
        expected_value: Optional[float] = None,
        adaptive_warning: Optional[str] = None,
    ) -> bool:
        """Append a decision record to the JSONL log file.

        Args:
            game_state: Snapshot of the raw game state before taking action.
            questions_asked: Formulated questions sent to Laya.
            probabilities_returned: Calibrated probabilities dictionary returned by Laya.
            chosen_action: The selected action dispatched to the game.
            compressed_state: Feature-engineered dense state string (Feature 1).
            expected_values: Dict mapping actions to their computed Expected Value scores (Feature 3).
            expected_value: Expected Value score of the chosen action (Feature 3).
            adaptive_warning: Failure warning appended to prompt if applicable (Feature 4).

        Returns:
            True if write succeeded, False otherwise.
        """
        # Derive chosen action's expected value if not explicitly passed
        if expected_value is None and expected_values and chosen_action in expected_values:
            expected_value = expected_values[chosen_action]

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "game_state": game_state,
            "compressed_state": compressed_state,
            "questions_asked": questions_asked,
            "probabilities_returned": probabilities_returned,
            "expected_values": expected_values or {},
            "expected_value": expected_value,
            "chosen_action": chosen_action,
            "adaptive_warning": adaptive_warning,
        }

        try:
            # Append mode with explicit UTF-8 encoding and newline flushing
            with open(self.log_path, mode="a", encoding="utf-8") as file_handle:
                file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                file_handle.flush()
            return True
        except (IOError, OSError) as exc:
            # Handle potential file lock issues or OS-level file contention gracefully
            print(f"[DecisionLogger Error] Failed to write decision log to {self.log_path}: {exc}")
            return False

    def read_recent_logs(self, limit: Optional[int] = 10) -> List[Dict[str, Any]]:
        """Safely read and parse recent log entries from the JSONL file.

        Handles incomplete/partially written lines during concurrent read/writes.

        Args:
            limit: Maximum number of recent entries to return (most recent last). None for all.

        Returns:
            List of parsed log entry dictionaries.
        """
        if not os.path.exists(self.log_path):
            return []

        records: List[Dict[str, Any]] = []
        try:
            with open(self.log_path, mode="r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                        records.append(record)
                    except json.JSONDecodeError:
                        # Skip partially written lines without crashing
                        continue
        except (IOError, OSError) as exc:
            print(f"[DecisionLogger Error] Failed to read log file {self.log_path}: {exc}")
            return []

        if limit is not None and limit > 0:
            return records[-limit:]
        return records
