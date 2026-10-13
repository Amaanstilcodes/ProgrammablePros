"""Laya Agent Module.

Interfaces with the Laya calibrated probability decision model.
Formulates typed questions per action candidate and retrieves calibrated success probabilities.
"""

import random
from typing import Any, Dict, List, Optional


class LayaAgent:
    """Agent that uses Laya probability calibration to evaluate candidate moves."""

    def __init__(self, api_key: Optional[str] = None, use_mock: bool = True) -> None:
        """Initialize the Laya Agent.

        Args:
            api_key: Optional API key for authenticating with Laya inference endpoint.
            use_mock: If True, uses synthetic calibrated probabilities for testing.
        """
        self.api_key: Optional[str] = api_key
        self.use_mock: bool = use_mock
        self.last_questions: Dict[str, str] = {}

    def formulate_questions(self, state: Dict[str, Any], possible_actions: List[str]) -> Dict[str, str]:
        """Convert game state and candidate actions into typed, evaluable questions for Laya.

        Laya does not output raw actions directly; it evaluates the probability of success
        for specific structured questions.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid action strings.

        Returns:
            Dict mapping action name to its corresponding typed question.
        """
        player_pos = state.get("player_pos", 0)
        goal_pos = state.get("goal_pos", 10)

        questions: Dict[str, str] = {}
        for action in possible_actions:
            questions[action] = (
                f"Given current player position {player_pos} and goal position {goal_pos}, "
                f"will moving '{action}' make progress toward the goal and result in a win?"
            )
        return questions

    def decide(self, state: Dict[str, Any], possible_actions: List[str]) -> Dict[str, float]:
        """Evaluate each candidate action with Laya and return calibrated probabilities.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid action strings.

        Returns:
            Dict mapping each candidate action to its calibrated success probability [0.0, 1.0].
        """
        # 1. Generate typed questions for each action
        self.last_questions = self.formulate_questions(state, possible_actions)

        # 2. Query model (Mock or Real Laya API)
        if self.use_mock:
            return self._mock_laya_eval(state, possible_actions)

        # -------------------------------------------------------------------------
        # TODO: REAL LAYA API INTEGRATION (Hackathon Day)
        # -------------------------------------------------------------------------
        # Once the official 'laya' pip package and credentials are provided:
        #
        # 1. Uncomment and import the official package:
        #    import laya
        #
        # 2. Instantiate the Laya client:
        #    client = laya.Client(api_key=self.api_key)
        #
        # 3. Submit typed questions for probability calibration:
        #    # Example API pattern:
        #    probabilities = {}
        #    for action, question in self.last_questions.items():
        #        response = client.predict_probability(
        #            context=f"Game State: {state}",
        #            question=question
        #        )
        #        # response.probability returns a calibrated float between 0.0 and 1.0
        #        probabilities[action] = round(float(response.probability), 4)
        #
        #    return probabilities
        # -------------------------------------------------------------------------
        raise NotImplementedError("Real Laya API mode selected, but SDK integration is pending hackathon credentials.")

    def _mock_laya_eval(self, state: Dict[str, Any], possible_actions: List[str]) -> Dict[str, float]:
        """Simulate calibrated probabilities for actions in mock environment.

        Generates realistic calibrated probabilities: moving toward the goal has
        a higher expected probability with slight stochastic variance.
        """
        player_pos = state.get("player_pos", 0)
        goal_pos = state.get("goal_pos", 10)

        raw_scores: Dict[str, float] = {}

        for action in possible_actions:
            if action == "right" and player_pos < goal_pos:
                # Advancing towards goal: calibrated probability typically between 0.65 - 0.95
                raw_scores[action] = round(random.uniform(0.65, 0.95), 3)
            elif action == "left":
                # Moving away from goal: lower calibrated probability between 0.05 - 0.35
                raw_scores[action] = round(random.uniform(0.05, 0.35), 3)
            else:
                raw_scores[action] = round(random.uniform(0.1, 0.5), 3)

        # Normalize across candidate actions so they represent relative choice confidence
        total_score = sum(raw_scores.values()) or 1.0
        normalized_probabilities: Dict[str, float] = {
            action: round(score / total_score, 3)
            for action, score in raw_scores.items()
        }

        return normalized_probabilities

    def get_last_questions(self) -> Dict[str, str]:
        """Retrieve the typed questions formulated during the most recent decision call."""
        return self.last_questions
