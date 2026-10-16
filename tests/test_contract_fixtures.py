"""Tests validating raw contract fixtures from TopGameAI Arena."""

from __future__ import annotations

from games.game2048.parser import Game2048StateParser
from games.snake.parser import SnakeStateParser


def test_snake_raw_fixture_parsing() -> None:
    raw_response = {
        "session_id": "SNK-10293",
        "state": {
            "snake": [[10, 8], [9, 8], [8, 8]],
            "food": [14, 2],
            "direction": "right",
            "score": 120,
            "step": 5,
        },
    }

    parser = SnakeStateParser()
    parsed = parser.parse(raw_response["state"])

    assert parsed["head"] == (10, 8)
    assert parsed["food"] == (14, 2)
    assert parsed["direction"] == "right"
    assert parsed["score"] == 120
    assert parsed["game_over"] is False


def test_2048_raw_fixture_parsing() -> None:
    raw_response = {
        "session_id": "2048-9921",
        "state": {
            "board": [
                [2, 0, 0, 0],
                [0, 4, 0, 0],
                [0, 0, 8, 0],
                [0, 0, 0, 16],
            ],
            "score": 30,
            "max_tile": 16,
            "done": False,
        },
    }

    parser = Game2048StateParser()
    parsed = parser.parse(raw_response["state"])

    assert parsed["max_tile"] == 16
    assert parsed["score"] == 30
    assert parsed["game_over"] is False
    assert parsed["board"][3][3] == 16
