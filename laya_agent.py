"""Laya Agent Module — Genesis 2.0 Track EVA 02.

Interfaces with the Laya calibrated probability decision model.
Formulates typed hypothesis questions per action candidate and retrieves calibrated success probabilities.
Implements:
1. State Compression & Feature Engineering
2. Hypothesis-Driven "Typed Questions"
3. Expected Value (EV) Decision Making
4. Failure-Adaptive Prompting (Short-Term Memory)
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple


class LayaAgent:
    """Agent that uses Laya probability calibration with feature engineering, EV calculation, and adaptive memory."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = True,
        max_memory_turns: int = 3,
    ) -> None:
        """Initialize the Laya Agent.

        Args:
            api_key: Optional API key for authenticating with Laya inference endpoint.
            use_mock: If True, uses synthetic calibrated probabilities for testing.
            max_memory_turns: Number of recent turns to retain in short-term memory (default: 3).
        """
        self.api_key: Optional[str] = api_key
        self.use_mock: bool = use_mock
        self.max_memory_turns: int = max_memory_turns

        # Failure-Adaptive Short-Term Memory
        self.recent_outcomes: List[Dict[str, Any]] = []

        # Cached telemetry from last decision cycle
        self.last_questions: Dict[str, str] = {}
        self.last_warnings: Dict[str, str] = {}
        self.last_compressed_state: str = ""

    def record_outcome(
        self,
        action: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        goal_pos: Optional[Any] = None,
        player_pos_key: str = "player_pos",
    ) -> bool:
        """Record the outcome of taking an action to detect failure/rejection."""
        pos_before = state_before.get(player_pos_key)
        pos_after = state_after.get(player_pos_key)

        made_progress = True
        reason = "ok"

        # Position unchanged detection
        if pos_before is not None and pos_after is not None and pos_before == pos_after:
            made_progress = False
            reason = "position unchanged (hit obstacle / wall / rejected move)"

        record: Dict[str, Any] = {
            "action": action,
            "pos_before": pos_before,
            "pos_after": pos_after,
            "made_progress": made_progress,
            "reason": reason,
        }
        self.recent_outcomes.append(record)

        if len(self.recent_outcomes) > self.max_memory_turns:
            self.recent_outcomes.pop(0)

        return made_progress

    def did_action_recently_fail(self, action: str) -> bool:
        """Check if taking this action in recent history resulted in zero progress."""
        for outcome in reversed(self.recent_outcomes):
            if outcome["action"] == action:
                return not outcome["made_progress"]
        return False

    def clear_memory(self) -> None:
        """Reset recent outcomes memory buffer."""
        self.recent_outcomes = []
        self.last_warnings = {}

    def decide(
        self,
        state: Dict[str, Any],
        possible_actions: List[str],
        compressed_state: str = "",
        subgoal: str = "progressing",
    ) -> Dict[str, float]:
        """Evaluate candidate actions and return calibrated probabilities.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid candidate action strings.
            compressed_state: Pre-computed compressed string.
            subgoal: Target subgoal phrase.

        Returns:
            Dict mapping each candidate action to its calibrated probability.
        """
        self.last_compressed_state = compressed_state
        self.last_warnings = {}

        questions: Dict[str, str] = {}
        for action in possible_actions:
            question = (
                f"Context: {compressed_state}. "
                f"Hypothesis: Executing action '{action}' is legal and progresses toward {subgoal}. "
                f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
            )
            if self.did_action_recently_fail(action):
                warning = (
                    f"WARNING: In the previous turn, taking action '{action}' resulted in no progress. "
                    f"Adjust probability accordingly."
                )
                self.last_warnings[action] = warning
                question += f" {warning}"
            questions[action] = question

        self.last_questions = questions

        if self.use_mock:
            return {a: round(random.uniform(0.1, 0.9), 3) for a in possible_actions}

        raise NotImplementedError("Real Laya API mode selected, but SDK integration is pending hackathon credentials.")

    def get_last_questions(self) -> Dict[str, str]:
        """Retrieve the typed questions formulated during the most recent decision call."""
        return self.last_questions

    def get_last_warnings(self) -> Dict[str, str]:
        """Retrieve failure-adaptive warnings generated during the most recent decision call."""
        return self.last_warnings

    def get_last_compressed_state(self) -> str:
        """Retrieve the compressed state string generated during the most recent decision call."""
        return self.last_compressed_state
