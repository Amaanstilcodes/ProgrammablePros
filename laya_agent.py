"""Laya Agent Module.

Interfaces with the Laya calibrated probability decision model.
Formulates typed hypothesis questions per action candidate and retrieves calibrated success probabilities.
Implements:
1. Feature 1: State Compression & Feature Engineering
2. Feature 2: Hypothesis-Driven "Typed Questions"
3. Feature 3: Expected Value (EV) Decision Making
4. Feature 4: Failure-Adaptive Prompting (Short-Term Memory)
"""

from typing import Any, Dict, List, Optional, Tuple
import random
import os


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

        Strategy 4: Keyed by (position, action) so a failure at one tile does NOT
        wrongly suppress the same action at a different position.

        Args:
            action: Action executed.
            prev_state: State before action execution.
            new_state: State after action execution.
            success: Whether the action achieved forward progress toward the goal.
            details: Human-readable diagnostic description of the transition.
        """
        prev_pos = prev_state.get("player_pos", 0)
        outcome: Dict[str, Any] = {
            "action": action,
            "prev_pos": prev_pos,
            "new_pos": new_state.get("player_pos", 0),
            "success": success,
            "details": details,
            "step": prev_state.get("current_step", 0),
        }
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > self.max_memory_turns:
            self.recent_outcomes.pop(0)

    def did_action_recently_fail(
        self, action: str, current_pos: Any = None
    ) -> bool:
        """Check if an action resulted in a failure within recent memory window.

        Strategy 4: When current_pos is provided (2-D Vault game), failure check is
        keyed by (position, action) so a hidden-wall at (3,2) does NOT suppress
        the same direction elsewhere on the grid.

        Args:
            action: Candidate action string to check.
            current_pos: Optional player position for position-scoped failure lookup.

        Returns:
            True if action failed in recent outcomes (at this position), False otherwise.
        """
        for outcome in reversed(self.recent_outcomes):
            if outcome.get("action") != action:
                continue
            if outcome.get("success", True):
                continue
            if current_pos is not None:
                # Position-scoped failure check (Strategy 4 for grid games)
                if outcome.get("prev_pos") == current_pos:
                    return True
            else:
                return True
        return False

    def clear_memory(self) -> None:
        """Reset short-term memory buffer (e.g., at game start/restart)."""
        self.recent_outcomes.clear()
        self.last_questions.clear()
        self.last_warnings.clear()
        self.last_compressed_state = ""

    def formulate_questions(
        self,
        state: Dict[str, Any],
        possible_actions: List[str],
        goal_phrase: str = "the goal",
        compressed_state_override: Optional[str] = None,
    ) -> Dict[str, str]:
        """Convert compressed state and actions into typed, hypothesis-driven questions (Feature 2 & 4).

        Strategy 2: goal_phrase parameter (default='the goal') lets Vault Runner
        inject a richer objective description without changing the original game.

        Frames binary, testable hypotheses:
        "Context: {compressed_state}. Hypothesis: Executing action '{action}' will
        successfully progress toward {goal_phrase} without being caught or hitting a
        hazard. What is the calibrated probability (0.0 to 1.0) that this is true?"

        Appends failure-adaptive warnings if the action recently failed at this position.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid candidate actions.
            goal_phrase: Goal description injected into hypothesis text (Strategy 2).
            compressed_state_override: Pre-computed compressed string (skips compress_state).

        Returns:
            Dict mapping action to its formulated typed question.
        """
        if compressed_state_override is not None:
            compressed = compressed_state_override
        else:
            compressed = self.compress_state(state)
        self.last_compressed_state = compressed
        self.last_warnings = {}

        current_pos = state.get("player_pos")  # None for 1-D game (unchanged behaviour)

        questions: Dict[str, str] = {}
        for action in possible_actions:
            question = (
                f"Context: {compressed}. "
                f"Hypothesis: Executing action '{action}' will successfully progress toward "
                f"{goal_phrase} without being caught or hitting a hazard. "
                f"What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"
            )

            # Feature 4: Failure-Adaptive Prompting (position-scoped for grid games)
            if self.did_action_recently_fail(action, current_pos=current_pos):
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
        payoffs: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """Compute Expected Value (EV) for each candidate action (Feature 3).

        EV = (P_success * Reward) - (P_failure * |Penalty|)

        Strategy 3: Optional `payoffs` dict maps action -> (reward, abs_penalty) for
        asymmetric per-action risk/reward (e.g. SPRINT vs WAIT in Vault Runner).
        When payoffs is None the global reward/penalty are used and the original
        game behaves exactly as before.

        Args:
            probabilities: Calibrated success probabilities per action.
            reward: Global payoff value (default: +10.0, used when payoffs=None).
            penalty: Global penalty score (default: 5.0, used when payoffs=None).
            payoffs: Optional per-action (reward, abs_penalty) overrides (Strategy 3).

        Returns:
            Dict mapping action to its calculated expected value score.
        """
        ev_scores: Dict[str, float] = {}
        for action, p_success in probabilities.items():
            if payoffs and action in payoffs:
                r, p = payoffs[action]   # Strategy 3: asymmetric payoff
            else:
                r, p = reward, penalty
            p_failure = max(0.0, 1.0 - p_success)
            ev = (p_success * r) - (p_failure * p)
            ev_scores[action] = round(ev, 3)
        return ev_scores

    def decide(
        self,
        state: Dict[str, Any],
        possible_actions: List[str],
        goal_phrase: str = "the goal",
        compressed_state_override: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate candidate actions and return calibrated probabilities.

        Args:
            state: Current game state dictionary.
            possible_actions: List of valid candidate action strings.
            goal_phrase: Passed through to formulate_questions (Strategy 2).
            compressed_state_override: Pre-computed compressed string (Strategy 1).

        Returns:
            Dict mapping each candidate action to its calibrated success probability.
        """
        # 1. Formulate hypothesis questions with state compression & failure warnings
        self.last_questions = self.formulate_questions(
            state,
            possible_actions,
            goal_phrase=goal_phrase,
            compressed_state_override=compressed_state_override,
        )

        # 2. Query model (Mock or Real Laya API)
        if self.use_mock:
            # Route to game-specific mock heuristic based on game_type field
            game_type = state.get("game_type", "CLASSIC")
            if game_type == "VAULT_RUNNER":
                return self._mock_vault_eval(state, possible_actions)
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

    def _mock_vault_eval(
        self, state: Dict[str, Any], possible_actions: List[str]
    ) -> Dict[str, float]:
        """Simulate calibrated probabilities for Vault Runner (offline/mock mode).

        Heuristic rules (Strategy 1 compressed fields drive the scoring):
        - Actions that reduce obj_dist get high P (0.60-0.90).
        - Actions toward a laser/guard get penalised P (0.15-0.35).
        - Actions in last_warnings (Strategy 4) get strongly suppressed (x0.15).
        - Small seeded noise added; all values clipped to [0.02, 0.98].
        """
        from vault_game import DIRECTION_DELTA, TILE_LASER, _manhattan  # local import

        px, py = state.get("player_pos", (0, 0))
        keys_held = state.get("keys_held", 0)
        guard_pos = state.get("guard_pos", (-1, -1))
        guard_next = state.get("guard_next_pos", (-1, -1))
        vgrid = state.get("visible_grid", [])

        # Determine current objective position
        if keys_held == 0:
            obj_pos = state.get("key_pos")
        else:
            obj_pos = state.get("goal_pos")

        raw_scores: Dict[str, float] = {}

        for action in possible_actions:
            has_warning = action in self.last_warnings

            if has_warning:
                # Strategy 4: suppress probability strongly when WARNING is active
                base = random.uniform(0.04, 0.14)
                raw_scores[action] = round(max(0.02, min(0.98, base)), 3)
                continue

            action_lc = action.strip().lower()

            if action_lc == "wait":
                base = random.uniform(0.30, 0.50)

            elif action_lc == "sprint":
                facing = state.get("facing", "right")
                if facing in DIRECTION_DELTA:
                    dx, dy = DIRECTION_DELTA[facing]
                    t1x, t1y = px + dx, py + dy
                    t2x, t2y = px + dx * 2, py + dy * 2
                    # Sprint toward objective => higher P
                    if obj_pos and _manhattan((t2x, t2y), obj_pos) < _manhattan((px, py), obj_pos):
                        base = random.uniform(0.65, 0.88)
                    else:
                        base = random.uniform(0.35, 0.60)
                    # Laser at tile 1: sprint jumps it safely, slightly boosts P
                    if vgrid and 0 <= t1x < len(vgrid[0]) if vgrid else False:
                        if 0 <= t1y < len(vgrid) and vgrid[t1y][t1x] == TILE_LASER:
                            base = min(0.98, base + 0.08)
                else:
                    base = random.uniform(0.30, 0.55)

            elif action_lc in DIRECTION_DELTA:
                dx, dy = DIRECTION_DELTA[action_lc]
                nx, ny = px + dx, py + dy

                # Guard adjacency: suppress strongly
                if (nx, ny) == guard_pos or (nx, ny) == guard_next:
                    base = random.uniform(0.05, 0.18)
                # Laser adjacency: suppress
                elif vgrid and 0 <= ny < len(vgrid) and 0 <= nx < len(vgrid[0] if vgrid else []):
                    if vgrid[ny][nx] == TILE_LASER:
                        base = random.uniform(0.15, 0.35)
                    elif obj_pos and _manhattan((nx, ny), obj_pos) < _manhattan((px, py), obj_pos):
                        # Advancing toward objective
                        base = random.uniform(0.62, 0.92)
                    else:
                        base = random.uniform(0.10, 0.40)
                elif obj_pos and _manhattan((nx, ny), obj_pos) < _manhattan((px, py), obj_pos):
                    base = random.uniform(0.62, 0.92)
                else:
                    base = random.uniform(0.10, 0.40)
            else:
                base = random.uniform(0.20, 0.50)

            # Small noise
            noise = random.uniform(-0.03, 0.03)
            raw_scores[action] = round(max(0.02, min(0.98, base + noise)), 3)

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
