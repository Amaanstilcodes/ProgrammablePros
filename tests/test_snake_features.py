"""Tests for Snake Feature Extraction and Egocentric Coordinates."""

from __future__ import annotations

from games.snake.features import SnakeFeatureExtractor
from games.snake.memory import SnakeMemory
from games.snake.parser import SnakeStateParser


def test_snake_features_bearing_and_obstacles() -> None:
    parser = SnakeStateParser()
    extractor = SnakeFeatureExtractor(grid_size=16)
    memory = SnakeMemory()

    raw_state = {
        "snake": [[5, 5], [5, 6], [5, 7]],
        "food": [8, 5],
        "direction": "up",
        "score": 0,
    }

    parsed = parser.parse(raw_state)
    features = extractor.extract_features(parsed, memory)

    assert features["player_pos"] == (5, 5)
    assert features["goal_pos"] == (8, 5)
    assert features["distance_to_goal"] == 3
    assert features["bearing"] == "E"
    assert features["facing"] == "up"

    # Moving backwards into body at (5, 6) is an obstacle
    assert features["obstacles"]["down"] is True
