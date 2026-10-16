"""Snake-Specific Decision Brain — Genesis 2.0 Track EVA 02.

Adapts the decision pipeline for Snake:
- SnakeCompressor: Compresses state into <= 200 char string
- SnakeSubgoalManager: Priority cascade (P1: SURVIVE, P2: CHASE_FOOD)
- SnakeHypothesisBuilder: 8 typed questions (4 directions x 2 questions)
- SnakeEVCalculator: Multi-term expected value with legality checks
"""

from __future__ import annotations

import math
from typing import Any


class SnakeCompressor:
    """Compresses Snake game state into a dense, high-signal prompt string (<= 200 chars)."""

    def __init__(self) -> None:
        self.last_failed_action: str | None = None

    def record_failure(self, action: str) -> None:
        self.last_failed_action = action

    def clear_failure(self) -> None:
        self.last_failed_action = None

    def compress(self, state: dict[str, Any]) -> str:
        snake = state.get("snake", [[0, 0]])
        head = snake[0] if snake else [0, 0]
        direction = state.get("direction", "unknown")
        body_len = len(snake)
        width = state.get("width", 20)
        height = state.get("height", 20)
        food = state.get("food", [0, 0])
        foods_eaten = state.get("foods_eaten", 0)
        step = state.get("step", 0)
        steps_rem = max(0, 300 - step)

        # Distances to walls: up, down, left, right
        d_up = head[1]
        d_down = height - 1 - head[1]
        d_left = head[0]
        d_right = width - 1 - head[0]

        # Manhattan distance to food
        m_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

        # Body segments adjacent to head
        body_set = {(s[0], s[1]) for s in snake[1:]}
        adj_ahead = 0
        adj_left = 0
        adj_right = 0
        if direction == "up":
            adj_ahead = int((head[0], head[1] - 1) in body_set)
            adj_left = int((head[0] - 1, head[1]) in body_set)
            adj_right = int((head[0] + 1, head[1]) in body_set)
        elif direction == "down":
            adj_ahead = int((head[0], head[1] + 1) in body_set)
            adj_left = int((head[0] + 1, head[1]) in body_set)
            adj_right = int((head[0] - 1, head[1]) in body_set)
        elif direction == "left":
            adj_ahead = int((head[0] - 1, head[1]) in body_set)
            adj_left = int((head[0], head[1] + 1) in body_set)
            adj_right = int((head[0], head[1] - 1) in body_set)
        elif direction == "right":
            adj_ahead = int((head[0] + 1, head[1]) in body_set)
            adj_left = int((head[0], head[1] - 1) in body_set)
            adj_right = int((head[0], head[1] + 1) in body_set)

        warn = f" W!{self.last_failed_action}" if self.last_failed_action else ""

        compact = (
            f"H:({head[0]},{head[1]}) D:{direction} L:{body_len} "
            f"F:({food[0]},{food[1]}) dF:{m_dist} "
            f"W:{d_up}/{d_down}/{d_left}/{d_right} "
            f"B:{adj_ahead}{adj_left}{adj_right} "
            f"E:{foods_eaten} S:{steps_rem}{warn}"
        )
        return compact[:200]


class SnakeSubgoalManager:
    """Deterministic priority cascade for Snake subgoals."""

    def get_current_subgoal(self, state: dict[str, Any]) -> str:
        snake = state.get("snake", [[0, 0]])
        head = snake[0] if snake else [0, 0]
        direction = state.get("direction", "right")
        width = state.get("width", 20)
        height = state.get("height", 20)

        # Danger ahead check: wall within 2 cells or body within 2 cells in current direction
        body_set = {(s[0], s[1]) for s in snake[1:]}
        near_hazard = False

        if direction == "up":
            if head[1] <= 2 or (head[0], head[1] - 1) in body_set or (head[0], head[1] - 2) in body_set:
                near_hazard = True
        elif direction == "down":
            if head[1] >= height - 3 or (head[0], head[1] + 1) in body_set or (head[0], head[1] + 2) in body_set:
                near_hazard = True
        elif direction == "left":
            if head[0] <= 2 or (head[0] - 1, head[1]) in body_set or (head[0] - 2, head[1]) in body_set:
                near_hazard = True
        elif direction == "right":
            if head[0] >= width - 3 or (head[0] + 1, head[1]) in body_set or (head[0] + 2, head[1]) in body_set:
                near_hazard = True

        if near_hazard:
            return "surviving and avoiding collision"
        return "reaching the food while staying safe"


class SnakeHypothesisBuilder:
    """Builds 8 typed hypothesis questions (4 directions x 2 questions)."""

    DIRECTIONS: list[str] = ["up", "down", "left", "right"]

    def build_hypotheses(self, state: dict[str, Any], subgoal: str, compressed: str) -> list[str]:
        questions: list[str] = []
        for action in self.DIRECTIONS:
            q1 = (
                f"Context: {compressed}. Hypothesis: Moving '{action}' is legal "
                f"and progresses toward {subgoal}. What is the calibrated "
                f"probability (0.0 to 1.0) that this is true?"
            )
            q2 = (
                f"Context: {compressed}. Hypothesis: Moving '{action}' causes "
                f"immediate collision with wall or body. What is the calibrated "
                f"probability (0.0 to 1.0) that this is true?"
            )
            questions.append(q1)
            questions.append(q2)
        return questions


class SnakeEVCalculator:
    """Calculates EV for each direction based on progress probability and danger probability."""

    OPPOSITE: dict[str, str] = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }

    DELTA: dict[str, tuple[int, int]] = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    def __init__(self) -> None:
        self.move_history: list[str] = []

    def record_move(self, action: str) -> None:
        self.move_history.append(action)
        if len(self.move_history) > 10:
            self.move_history.pop(0)

    def calculate(
        self,
        state: dict[str, Any],
        raw_probs: dict[str, dict[str, float]],
        subgoal: str,
    ) -> tuple[str, dict[str, Any]]:
        """Calculate EV per direction.

        EV(a) = P_progress(a) * 10.0 - P_danger(a) * 40.0 - repeat_penalty(a) - illegal_penalty(a)
        """
        snake = state.get("snake", [[0, 0]])
        head = snake[0] if snake else [0, 0]
        current_dir = state.get("direction", "")
        width = state.get("width", 20)
        height = state.get("height", 20)
        body_excluding_tail = snake[:-1] if len(snake) > 1 else snake

        ev_breakdown: dict[str, Any] = {}
        best_action = "up"
        best_ev = -1e9

        for action in ("up", "down", "left", "right"):
            probs = raw_probs.get(action, {"q1": 0.25, "q2": 0.25})
            p_progress = float(probs.get("q1", 0.25))
            p_danger = float(probs.get("q2", 0.25))

            # Repeat penalty: +5.0 if action is same as last 2 moves
            repeat_penalty = 0.0
            if len(self.move_history) >= 2 and self.move_history[-1] == action and self.move_history[-2] == action:
                repeat_penalty = 5.0

            # Illegal penalty: +100.0 if reversal, wall, or body collision
            illegal_penalty = 0.0
            dx, dy = self.DELTA[action]
            nx, ny = head[0] + dx, head[1] + dy

            is_reversal = (len(snake) > 1 and action == self.OPPOSITE.get(current_dir))
            hits_wall = not (0 <= nx < width and 0 <= ny < height)
            hits_body = [nx, ny] in body_excluding_tail

            if is_reversal or hits_wall or hits_body:
                illegal_penalty = 100.0

            ev = (p_progress * 10.0) - (p_danger * 40.0) - repeat_penalty - illegal_penalty

            ev_breakdown[action] = {
                "ev": round(ev, 3),
                "p_progress": round(p_progress, 4),
                "p_danger": round(p_danger, 4),
                "repeat_penalty": repeat_penalty,
                "illegal_penalty": illegal_penalty,
            }

            if ev > best_ev:
                best_ev = ev
                best_action = action

        return best_action, ev_breakdown
