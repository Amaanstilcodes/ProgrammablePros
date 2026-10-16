"""Tests validating contract fixtures from TopGameAI Arena.

If captured fixtures do not exist under docs/fixtures/, tests skip with a clear message.
Also asserts that parsers never mutate incoming raw state.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.protocols import GameStateView
from games.game2048.parser import Game2048StateParser
from games.snake.parser import SnakeStateParser


def test_snake_parser_purity_and_view() -> None:
    """Verify that SnakeStateParser produces an immutable GameStateView and never mutates raw state."""
    raw_response = {
        "session_id": "SNK-10293",
        "snake": [[10, 8], [9, 8], [8, 8]],
        "food": [14, 2],
        "direction": "right",
        "score": 120,
        "step": 5,
        "game_over": False,
    }
    raw_copy = copy.deepcopy(raw_response)

    parser = SnakeStateParser()
    parser.validate(raw_response)
    view = parser.parse(raw_response, session_id="SNK-10293")

    # Assert purity: input was NOT mutated
    assert raw_response == raw_copy
    # Assert GameStateView output
    assert isinstance(view, GameStateView)
    assert view.head == (10, 8)
    assert view.food == (14, 2)
    assert view.direction == "right"
    assert view.score == 120
    assert view.game_over is False


def test_2048_parser_purity_and_view() -> None:
    """Verify that Game2048StateParser produces an immutable GameStateView and never mutates raw state."""
    raw_response = {
        "session_id": "2048-9921",
        "board": [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 16],
        ],
        "score": 30,
        "max_tile": 16,
        "game_over": False,
    }
    raw_copy = copy.deepcopy(raw_response)

    parser = Game2048StateParser()
    parser.validate(raw_response)
    view = parser.parse(raw_response, session_id="2048-9921")

    # Assert purity
    assert raw_response == raw_copy
    # Assert GameStateView output
    assert isinstance(view, GameStateView)
    assert view.best_tile == 16
    assert view.score == 30
    assert view.game_over is False
    assert view.board is not None
    assert view.board[3][3] == 16


def test_captured_disk_fixtures_snake() -> None:
    """Load real captured fixtures under docs/fixtures/snake/ if present, or skip."""
    fixture_dir = Path("docs/fixtures/snake")
    fixture_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
    # Ignore request metadata files
    state_files = [f for f in fixture_files if not f.name.endswith(".request.json")]

    if not state_files:
        pytest.skip(
            "SKIP: No captured Snake fixtures found in docs/fixtures/snake/. "
            "Run tools/probe_api.py --game snake to capture real arena responses."
        )

    parser = SnakeStateParser()
    for fpath in state_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_state = data.get("state", data)
        raw_before = copy.deepcopy(raw_state)

        parser.validate(raw_state)
        view = parser.parse(raw_state)

        assert isinstance(view, GameStateView)
        assert raw_state == raw_before, f"Mutation detected in {fpath}"


def test_captured_disk_fixtures_2048() -> None:
    """Load real captured fixtures under docs/fixtures/2048/ if present, or skip."""
    fixture_dir = Path("docs/fixtures/2048")
    fixture_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
    state_files = [f for f in fixture_files if not f.name.endswith(".request.json")]

    if not state_files:
        pytest.skip(
            "SKIP: No captured 2048 fixtures found in docs/fixtures/2048/. "
            "Run tools/probe_api.py --game 2048 to capture real arena responses."
        )

    parser = Game2048StateParser()
    for fpath in state_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_state = data.get("state", data)
        raw_before = copy.deepcopy(raw_state)

        parser.validate(raw_state)
        view = parser.parse(raw_state)

        assert isinstance(view, GameStateView)
        assert raw_state == raw_before, f"Mutation detected in {fpath}"
