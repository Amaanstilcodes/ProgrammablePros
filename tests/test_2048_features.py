"""Tests for 2048 Feature Extraction."""

from __future__ import annotations

from games.game2048.features import Game2048FeatureExtractor
from games.game2048.memory import Game2048Memory
from games.game2048.parser import Game2048StateParser


def test_2048_feature_extraction() -> None:
    parser = Game2048StateParser()
    extractor = Game2048FeatureExtractor()
    memory = Game2048Memory()

    raw_state = {
        "board": [
            [256, 128, 64, 32],
            [2, 4, 8, 16],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        "score": 1024,
    }

    parsed = parser.parse(raw_state)
    features = extractor.extract_features(parsed, memory)

    assert features["max_tile"] == 256
    assert features["is_corner"] is True  # Top-left corner (0, 0)
    assert features["empty_cells"] == 7
    assert features["available_merges"] >= 1  # 2 at (1,0) and (2,0)
