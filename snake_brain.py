"""snake_brain.py — Snake Perception, Hypothesis, and EV Pipeline — Genesis 2.0 Track EVA 02.

HACKATHON DAY PRODUCTION IMPLEMENTATION — ZERO MOCKS ALLOWED.
- SnakeCompressor: Compresses state into <= 200 char dense representation with Manhattan distance to food.
- SnakeHypothesisBuilder: Formulates noul typed questions explicitly mentioning food coordinates.
- SnakeEVCalculator: Multi-term Expected Value optimizing calibrated Laya probabilities and food acquisition:
    EV = (P_progress * 20.0) - (P_danger * 40.0) + food_bonus - illegal_penalty - repeat_penalty
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _parse_coord(val: Any, default: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    """Extract (x, y) integer tuple from coordinate dict {'x':..,'y':..} or list/tuple."""
    if isinstance(val, dict):
        return int(val.get("x", default[0])), int(val.get("y", default[1]))
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return int(val[0]), int(val[1])
    return default


def _parse_snake(raw_snake: Any) -> List[Tuple[int, int]]:
    """Extract list of (x, y) body coordinates with head at index 0."""
    coords: List[Tuple[int, int]] = []
    if isinstance(raw_snake, (list, tuple)):
        for item in raw_snake:
            coords.append(_parse_coord(item))
    return coords or [(0, 0)]


def parse_snake_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse real TopGameAI Snake arena state into standardized fields."""
    raw_snake = state.get("snake", [])
    snake_coords = _parse_snake(raw_snake)
    food_coord = _parse_coord(state.get("food", [0, 0]))
    direction = str(state.get("direction", "right")).lower()
    width = int(state.get("width", 12))
    height = int(state.get("height", 12))
    foods_eaten = int(state.get("foodEaten", state.get("foods_eaten", 0)))
    step = int(state.get("steps", state.get("step", 0)))
    score = int(state.get("score", foods_eaten * 60))
    game_over = bool(state.get("done", state.get("game_over", False)))
    won = bool(state.get("won", False))

    return {
        "snake": snake_coords,
        "head": snake_coords[0] if snake_coords else (0, 0),
        "food": food_coord,
        "direction": direction,
        "width": width,
        "height": height,
        "foods_eaten": foods_eaten,
        "foodEaten": foods_eaten,
        "step": step,
        "steps": step,
        "score": score,
        "game_over": game_over,
        "done": game_over,
        "won": won,
        "grid": state.get("grid"),
    }


class SnakeCompressor:
    """Compresses Snake game state into a dense, high-signal representation (<= 200 chars).

    Format:
    H:(x,y) D:dir F:(fx,fy) dF:dist W:up/down/left/right B:ahead/left/right E:foods_eaten
    """

    def __init__(self) -> None:
        self.last_failed_action: str | None = None

    def record_failure(self, action: str) -> None:
        self.last_failed_action = action

    def clear_failure(self) -> None:
        self.last_failed_action = None

    def compress(self, state: Dict[str, Any]) -> str:
        parsed = parse_snake_state(state)
        snake = parsed["snake"]
        head = parsed["head"]
        direction = parsed["direction"]
        width = parsed["width"]
        height = parsed["height"]
        food = parsed["food"]
        foods_eaten = parsed["foods_eaten"]

        # Manhattan distance to food
        m_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

        # Distances to walls: up, down, left, right
        d_up = head[1]
        d_down = height - 1 - head[1]
        d_left = head[0]
        d_right = width - 1 - head[0]
        w_str = f"{d_up}/{d_down}/{d_left}/{d_right}"

        # Body segments adjacent to head relative to direction
        body_set = {(s[0], s[1]) for s in snake[1:]}
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
        else:  # right
            adj_ahead = int((head[0] + 1, head[1]) in body_set)
            adj_left = int((head[0], head[1] - 1) in body_set)
            adj_right = int((head[0], head[1] + 1) in body_set)

        b_str = f"{adj_ahead}/{adj_left}/{adj_right}"

        compact = (
            f"H:({head[0]},{head[1]}) D:{direction} "
            f"F:({food[0]},{food[1]}) dF:{m_dist} "
            f"W:{w_str} B:{b_str} E:{foods_eaten}"
        )
        return compact[:200]


class SnakeHypothesisBuilder:
    """Builds noul typed questions explicitly mentioning food coordinates to guide Laya."""

    DIRECTIONS: List[str] = ["up", "down", "left", "right"]

    def build_hypotheses(
        self,
        state: Dict[str, Any],
        subgoal: Any = None,
        compressed: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Return questions for Laya model scoring."""
        parsed = parse_snake_state(state)
        food_pos = parsed["food"]

        if subgoal is not None and isinstance(subgoal, str) and compressed:
            # 8-question list format (2 per action: progress and danger) for run_local_games
            questions: List[str] = []
            for action in self.DIRECTIONS:
                q1 = f"Moving '{action}' is legal and progresses toward {subgoal}."
                q2 = f"Moving '{action}' causes an immediate collision with a wall or snake body."
                questions.extend([q1, q2])
            return questions

        questions_dict: Dict[str, Dict[str, Any]] = {}
        for action in self.DIRECTIONS:
            questions_dict[f"q_{action}"] = {
                "type": "noul",
                "instructions": (
                    f"Moving '{action}' is legal and progresses toward the food at {food_pos} while staying safe."
                ),
            }
        return questions_dict


    def build_hypothesis_strings(self, state: Dict[str, Any], compressed: str = "") -> List[str]:
        """Return list of question strings for official logging compliance."""
        parsed = parse_snake_state(state)
        food_pos = parsed["food"]
        return [
            f"Moving '{action}' is legal and progresses toward the food at {food_pos} while staying safe."
            for action in self.DIRECTIONS
        ]


class SnakeSubgoalManager:
    """Deterministic priority cascade for Snake subgoals."""

    def get_current_subgoal(self, state: Dict[str, Any]) -> str:
        parsed = parse_snake_state(state)
        snake = parsed["snake"]
        head = parsed["head"]
        direction = parsed["direction"]
        width = parsed["width"]
        height = parsed["height"]
        food = parsed["food"]

        # Check if immediate hazard in travel direction
        body_set = {(s[0], s[1]) for s in snake[1:]}
        near_hazard = False

        if direction == "up":
            if head[1] <= 1 or (head[0], head[1] - 1) in body_set:
                near_hazard = True
        elif direction == "down":
            if head[1] >= height - 2 or (head[0], head[1] + 1) in body_set:
                near_hazard = True
        elif direction == "left":
            if head[0] <= 1 or (head[0] - 1, head[1]) in body_set:
                near_hazard = True
        elif direction == "right":
            if head[0] >= width - 2 or (head[0] + 1, head[1]) in body_set:
                near_hazard = True

        if near_hazard:
            return "surviving and avoiding immediate collision"
        return f"reaching the food at {food} while staying safe"


class SnakeEVCalculator:
    """Calculates Expected Value (EV) per candidate action to optimize food acquisition and survival.

    Formula:
      EV = (P_progress * 20.0) - (P_danger * 40.0) + food_bonus - illegal_penalty - repeat_penalty

    Where:
      - food_bonus: +10.0 if the move reduces Manhattan distance to food, -5.0 if it increases distance
      - illegal_penalty: +100.0 if the move causes reversal, wall collision, or body collision
      - repeat_penalty: +5.0 if action is repeating excessively in loops
    """

    OPPOSITE: Dict[str, str] = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }

    DELTA: Dict[str, Tuple[int, int]] = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    def __init__(self) -> None:
        self.move_history: List[str] = []

    def record_move(self, action: str) -> None:
        self.move_history.append(action)
        if len(self.move_history) > 10:
            self.move_history.pop(0)

    def calculate(
        self,
        state: Dict[str, Any],
        raw_probs: Dict[str, Any],
        subgoal: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Calculate Expected Value for each direction.

        Args:
            state: Game state dictionary.
            raw_probs: Dict of action probabilities from Laya, e.g. {"up": 0.85, ...} or {"up": {"q1": 0.85, "q2": 0.05}}.
            subgoal: Optional subgoal context.

        Returns:
            Tuple of (best_action, ev_breakdown_dict).
        """
        parsed = parse_snake_state(state)
        snake = parsed["snake"]
        head = parsed["head"]
        food = parsed["food"]
        current_dir = parsed["direction"]
        width = parsed["width"]
        height = parsed["height"]

        # Body excluding tail (tail will advance unless snake just ate food)
        body_excluding_tail = set(snake[1:-1] if len(snake) > 2 else snake[1:])

        # Current Manhattan distance to food
        cur_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

        ev_breakdown: Dict[str, Any] = {}
        best_action = "up"
        best_ev = -1e9

        for action in ("up", "down", "left", "right"):
            prob_val = raw_probs.get(action, 0.25)
            if isinstance(prob_val, dict):
                p_progress = float(prob_val.get("progress", prob_val.get("q1", 0.25)))
                p_danger_raw = prob_val.get("danger", prob_val.get("q2"))
                p_danger = float(p_danger_raw) if p_danger_raw is not None else None
            else:
                p_progress = float(prob_val)
                p_danger = None

            dx, dy = self.DELTA[action]
            nx, ny = head[0] + dx, head[1] + dy

            # 1. Legality checks
            is_reversal = (len(snake) > 1 and action == self.OPPOSITE.get(current_dir))
            hits_wall = not (0 <= nx < width and 0 <= ny < height)
            hits_body = (nx, ny) in body_excluding_tail

            illegal_penalty = 100.0 if (is_reversal or hits_wall or hits_body) else 0.0

            # 2. Food-seeking bonus: reward moving closer to food
            new_dist = abs(nx - food[0]) + abs(ny - food[1])
            if new_dist < cur_dist:
                food_bonus = 10.0
            elif new_dist > cur_dist:
                food_bonus = -5.0
            else:
                food_bonus = 0.0

            # 3. Danger probability computation
            if is_reversal or hits_wall or hits_body:
                computed_danger = 1.0
            else:
                # Count adjacent hazards to detect dead ends / narrow corridors
                adj_hazards = 0
                for ox, oy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    adj_x, adj_y = nx + ox, ny + oy
                    if not (0 <= adj_x < width and 0 <= adj_y < height) or (adj_x, adj_y) in body_excluding_tail:
                        adj_hazards += 1

                if adj_hazards >= 3:
                    computed_danger = 0.8
                elif adj_hazards == 2:
                    computed_danger = 0.3
                else:
                    computed_danger = 0.05

            final_p_danger = p_danger if p_danger is not None else computed_danger

            # 4. Repeat penalty to discourage oscillation loops
            repeat_penalty = 0.0
            if len(self.move_history) >= 2 and self.move_history[-1] == action and self.move_history[-2] == action:
                repeat_penalty = 5.0

            # 5. Full EV Formula (PDF §3.3 + Hackathon Day Directives)
            ev = (p_progress * 20.0) - (final_p_danger * 40.0) + food_bonus - illegal_penalty - repeat_penalty

            ev_breakdown[action] = {
                "ev": round(ev, 3),
                "p_progress": round(p_progress, 4),
                "p_danger": round(final_p_danger, 4),
                "food_bonus": food_bonus,
                "illegal_penalty": illegal_penalty,
                "repeat_penalty": repeat_penalty,
            }

            if ev > best_ev:
                best_ev = ev
                best_action = action

        return best_action, ev_breakdown


__all__ = [
    "SnakeCompressor",
    "SnakeSubgoalManager",
    "SnakeHypothesisBuilder",
    "SnakeEVCalculator",
    "parse_snake_state",
]
