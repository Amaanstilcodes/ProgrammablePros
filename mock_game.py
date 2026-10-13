"""Mock Game Module.

Simulates a hosted online game environment.
Provides game state inspection and action dispatch methods mimicking an external API.
"""

from typing import Any, Dict, List


class MockGame:
    """A mock 1D grid navigation game where a player moves towards a goal."""

    def __init__(self, start_pos: int = 0, goal_pos: int = 10, max_steps: int = 30) -> None:
        """Initialize mock game environment.

        Args:
            start_pos: Initial position of the player (default: 0).
            goal_pos: Target position the player must reach to win (default: 10).
            max_steps: Maximum allowed turns before ending the game.
        """
        self.start_pos: int = start_pos
        self.player_pos: int = start_pos
        self.goal_pos: int = goal_pos
        self.max_steps: int = max_steps
        self.current_step: int = 0
        self.is_game_over: bool = False
        self.status: str = "IN_PROGRESS"  # "IN_PROGRESS", "WON", "MAX_STEPS_REACHED"

    def get_state(self) -> Dict[str, Any]:
        """Fetch the current state of the game.

        In a production scenario, this would be an HTTP GET request to the hosted game API.

        Returns:
            Dict containing player position, goal position, available actions, and game status.
        """
        return {
            "player_pos": self.player_pos,
            "goal_pos": self.goal_pos,
            "available_actions": ["left", "right"],
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "is_game_over": self.is_game_over,
            "status": self.status,
        }

    def send_action(self, action: str) -> Dict[str, Any]:
        """Send an action to the game and transition to the next state.

        In a production scenario, this would be an HTTP POST request to the hosted game API.

        Args:
            action: The selected action (e.g., 'left' or 'right').

        Returns:
            Dict containing the updated state after the action is processed.
        """
        if self.is_game_over:
            return self.get_state()

        self.current_step += 1
        normalized_action = action.strip().lower()

        if normalized_action == "right":
            self.player_pos += 1
        elif normalized_action == "left":
            self.player_pos = max(0, self.player_pos - 1)
        else:
            # Handle unknown action gracefully without crashing
            pass

        # Check win/termination conditions
        if self.player_pos >= self.goal_pos:
            self.is_game_over = True
            self.status = "WON"
        elif self.current_step >= self.max_steps:
            self.is_game_over = True
            self.status = "MAX_STEPS_REACHED"

        return self.get_state()

    def reset(self) -> Dict[str, Any]:
        """Reset the game back to the starting conditions."""
        self.player_pos = self.start_pos
        self.current_step = 0
        self.is_game_over = False
        self.status = "IN_PROGRESS"
        return self.get_state()
