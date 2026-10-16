"""Tests for Streamlit Viewer Parsers and ASCII Visualizers."""

from __future__ import annotations

from viewer_compliance import render_2048_html, render_snake_ascii


def test_render_snake_ascii() -> None:
    state = {
        "snake": [[2, 2], [2, 3]],
        "food": [5, 5],
    }
    ascii_grid = render_snake_ascii(state, width=10, height=10)
    assert "◆ " in ascii_grid
    assert "■ " in ascii_grid
    assert "★ " in ascii_grid


def test_render_2048_html() -> None:
    board = [
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 0],
        [0, 0, 0, 0],
    ]
    html = render_2048_html(board)
    assert '<div class="tile-grid">' in html
    assert "tile-2048" in html
    assert "2048" in html
