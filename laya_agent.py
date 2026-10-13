"""Laya Agent Module.

Interfaces with the Laya calibrated probability decision model.
Formulates typed hypothesis questions per action candidate and retrieves calibrated success probabilities.
Implements:
1. Feature 1: State Compression & Feature Engineering
2. Feature 2: Hypothesis-Driven "Typed Questions"
3. Feature 3: Expected Value (EV) Decision Making
4. Feature 4: Failure-Adaptive Prompting (Short-Term Memory)
"""

from typing import Any, Dict, List, Optional
import random


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

        # Feature 4: Failure-Adaptive Short-Term Memory
        self.recent_outcomes: List[Dict[str, Any]] = []

        # Cached telemetry from last decision cycle
        self.last_questions: Dict[str, str] = {}
        self.last_warnings: Dict[str, str] = {}
        self.last_compressed_state: str = ""

    def compress_state(self, state: Dict[str, Any]) -> str:
        """Compress raw game state dictionary into dense, high-signal prompt context (Feature 1).

        Small models get confused by noisy, raw JSON. This extracts only the 3-4 vital variables:
        player position, goal position, distance to goal, and steps remaining.

        Args:
            state: Full game state dictionary.

        Returns:
            Dense, clean string representation of vital state features.
        """
        player_pos = state.get("player_pos", 0)
        goal_pos = state.get("goal_pos", 10)
        max_steps = state.get("max_steps", 30)
        current_step = state.get("current_step", 0)

        distance_to_goal = abs(goal_pos - player_pos)
        steps_remaining = max(0, max_steps - current_step)

        return (
            f"player_pos={player_pos}, goal_pos={goal_pos}, "
            f"distance_to_goal={distance_to_goal}, steps_remaining={steps_remaining}"
        )

    def record_outcome(
        self,
        action: str,
        prev_state: Dict[str, Any],
        new_state: Dict[str, Any],
        success: bool,
        details: str = "",
    ) -> None:
        """Record an action outcome in short-term memory (Feature 4).

        Args:
            action: Action executed.
            prev_state: State before action execution.
            new_state: State after action execution.
            success: Whether the action achieved forward progress toward the goal.
            details: Human-readable diagnostic description of the transition.
        """
        outcome: Dict[str, Any] = {
            "action": action,
            "prev_pos": prev_state.get("player_pos", 0),
            "new_pos": new_state.get("player_pos", 0),
            "success": success,
            "details": details,
            "step": prev_state.get("current_step", 0),
        }
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > self.max_memory_turns:
            self.recent_outcomes.pop(0)

    def did_action_recently_fail(self, action: str) -> bool:
        """Check if an action resulted in a failure within recent memory window.

        Args:
            action: Candidate action string to check.

        Returns:
            True if action failed in recent outcomes, False otherwise.
        """
        for outcome in reversed(self.recent_outcomes):
            if outcome.get("action") == action and not outcome.get("success", True):
                return True
        return False

    def clear_memory(self) -> None:
        """Reset short-term memory buffer (e.g., at game start/restart)."""
        self.recent_outcomes.clear()
        self.last_questions.clear()
        self.last_warnings.clear()
        self.last_compressed_state = ""

    def formulate_questions(self, state: Dict[str, Any], possible_actions: List[str]) -> Dict[str, str]:
        """Convert compressed state and actions into typed, hypothesis-driven questions (Feature 2 & 4).

        Frames binary, testable hypotheses for probability calibration:
        "Context: {compressed_state}. Hypothesis: Executing action '{action}' will successfully progress toward the goal. What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
        Appends failure-adaptive warnings if the action recently failed.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid candidate actions.

        Returns:
            Dict mapping action to its formulated typed question.
        """
        compressed = self.compress_state(state)
        self.last_compressed_state = compressed
        self.last_warnings = {}

        questions: Dict[str, str] = {}
        for action in possible_actions:
            question = (
                f"Context: {compressed}. "
                f"Hypothesis: Executing action '{action}' will successfully progress toward the goal. "
                f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
            )

            # Feature 4: Failure-Adaptive Prompting
            if self.did_action_recently_fail(action):
                warning = (
                    f"WARNING: In the previous turn, taking action '{action}' resulted in no progress. "
                    f"Adjust probability accordingly."
                )
                self.last_warnings[action] = warning
                question += f" {warning}"

            questions[action] = question

        self.last_questions = questions
        return questions

    def calculate_expected_values(
        self,
        probabilities: Dict[str, float],
        reward: float = 10.0,
        penalty: float = 5.0,
    ) -> Dict[str, float]:
        """Compute Expected Value (EV) for each candidate action (Feature 3).

        EV = (Probability_of_Success * Reward) - (Probability_of_Failure * Penalty)
        where Probability_of_Failure = 1.0 - Probability_of_Success.

        Args:
            probabilities: Calibrated success probabilities per action.
            reward: Payoff value for successful progress (default: +10.0).
            penalty: Penalty score for failed move or wasted turn (default: 5.0).

        Returns:
            Dict mapping action to its calculated expected value score.
        """
        ev_scores: Dict[str, float] = {}
        for action, p_success in probabilities.items():
            p_failure = max(0.0, 1.0 - p_success)
            ev = (p_success * reward) - (p_failure * penalty)
            ev_scores[action] = round(ev, 3)
        return ev_scores

    def decide(self, state: Dict[str, Any], possible_actions: List[str]) -> Dict[str, float]:
        """Evaluate candidate actions and return calibrated probabilities.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid candidate action strings.

        Returns:
            Dict mapping each candidate action to its calibrated success probability.
        """
        # 1. Formulate hypothesis questions with state compression & failure warnings
        self.last_questions = self.formulate_questions(state, possible_actions)

        # 2. Query model (Mock or Real Laya API)
        if self.use_mock:
            return self._mock_laya_eval(state, possible_actions)

        # -------------------------------------------------------------------------
        # REAL LAYA API INTEGRATION (Hackathon Day)
        # -------------------------------------------------------------------------
        # When official 'laya' SDK and credentials are provided:
        #
        # import laya
        # client = laya.Client(api_key=self.api_key)
        # probabilities = {}
        # for action, question in self.last_questions.items():
        #     response = client.predict_probability(
        #         context=self.last_compressed_state,
        #         question=question
        #     )
        #     probabilities[action] = round(float(response.probability), 4)
        # return probabilities
        # -------------------------------------------------------------------------
        raise NotImplementedError("Real Laya API mode selected, but SDK integration is pending hackathon credentials.")

    def _mock_laya_eval(self, state: Dict[str, Any], possible_actions: List[str]) -> Dict[str, float]:
        """Simulate calibrated probabilities with failure-adaptive awareness.

        Behaviors:
        - Actions flagged by Failure-Adaptive warning receive sharply penalized probabilities.
        - Moving toward the goal yields high calibrated success probabilities (0.70 - 0.95).
        - Moving away from the goal yields low calibrated success probabilities (0.05 - 0.30).
        """
        player_pos = state.get("player_pos", 0)
        goal_pos = state.get("goal_pos", 10)

        raw_scores: Dict[str, float] = {}

        for action in possible_actions:
            # Check if this action is flagged with an active failure warning
            has_failure_warning = action in self.last_warnings

            if has_failure_warning:
                # Mock Laya heeds failure warning: drastically suppress calibrated probability
                raw_scores[action] = round(random.uniform(0.05, 0.15), 3)
            elif action == "right" and player_pos < goal_pos:
                # Advancing towards goal
                raw_scores[action] = round(random.uniform(0.70, 0.95), 3)
            elif action == "left" and player_pos > goal_pos:
                # Advancing towards goal in reverse navigation
                raw_scores[action] = round(random.uniform(0.70, 0.95), 3)
            elif action == "left":
                # Moving away from goal
                raw_scores[action] = round(random.uniform(0.05, 0.30), 3)
            else:
                raw_scores[action] = round(random.uniform(0.10, 0.45), 3)

        return raw_scores

    def get_last_questions(self) -> Dict[str, str]:
        """Retrieve the typed questions formulated during the most recent decision call."""
        return self.last_questions

    def get_last_warnings(self) -> Dict[str, str]:
        """Retrieve failure-adaptive warnings generated during the most recent decision call."""
        return self.last_warnings

    def get_last_compressed_state(self) -> str:
        """Retrieve the compressed state string generated during the most recent decision call."""
        return self.last_compressed_state
