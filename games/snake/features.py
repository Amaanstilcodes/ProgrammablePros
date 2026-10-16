"""Snake Feature Extractor — Genesis 2.0 Track EVA 02.

Computes egocentric spatial features for Snake:
- Manhattan distance and compass bearing to food
- Immediate obstacles (ahead, left, right) relative to current heading
- Confined pocket / dead-end risk assessment
"""

from __future__ import annotations

from typing import Any

from games.snake.parser import SnakeStateParser

DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

RELATIVE_TURNS: dict[str, dict[str, str]] = {
    "up": {"ahead": "up", "left": "left", "right": "right", "behind": "down"},
    "down": {"ahead": "down", "left": "right", "right": "left", "behind": "up"},
    "left": {"ahead": "left", "left": "down", "right": "up", "behind": "right"},
    "right": {"ahead": "right", "left": "up", "right": "down", "behind": "left"},
}


class SnakeFeatureExtractor:
    """Extracts deterministic spatial feature vectors from parsed Snake state."""

    def __init__(self, grid_size: int = 16) -> None:
        self.grid_size = grid_size

    def extract_features(
        self,
        parsed_state: dict[str, Any],
        memory: Any,
    ) -> dict[str, Any]:
        head: tuple[int, int] = parsed_state["head"]
        food: tuple[int, int] = parsed_state["food"]
        snake_body: list[tuple[int, int]] = parsed_state["snake"]
        direction: str = parsed_state["direction"]

        dx = food[0] - head[0]
        dy = food[1] - head[1]
        distance = abs(dx) + abs(dy)

        # Compass bearing to food
        bearing = ""
        if dy < 0:
            bearing += "N"
        elif dy > 0:
            bearing += "S"
        if dx > 0:
            bearing += "E"
        elif dx < 0:
            bearing += "W"
        if not bearing:
            bearing = "HERE"

        # Surrounding cell obstacles
        body_set = set(snake_body[:-1]) if len(snake_body) > 1 else set(snake_body)
        obstacles: dict[str, bool] = {}
        for action, delta in DELTAS.items():
            nx = head[0] + delta[0]
            ny = head[1] + delta[1]
            is_wall = nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size
            is_body = (nx, ny) in body_set
            obstacles[action] = is_wall or is_body

        # Egocentric POV mapping
        pov_mapping = RELATIVE_TURNS.get(direction, RELATIVE_TURNS["right"])
        egocentric_local = {
            "AHEAD": "WALL" if obstacles.get(pov_mapping["ahead"], False) else "FREE",
            "LEFT": "WALL" if obstacles.get(pov_mapping["left"], False) else "FREE",
            "RIGHT": "WALL" if obstacles.get(pov_mapping["right"], False) else "FREE",
            "BEHIND": "WALL" if obstacles.get(pov_mapping["behind"], False) else "FREE",
        }

        return {
            "player_pos": head,
            "goal_pos": food,
            "distance_to_goal": distance,
            "bearing": bearing,
            "facing": direction,
            "obstacles": obstacles,
            "egocentric_local": egocentric_local,
            "snake_length": len(snake_body),
        }
