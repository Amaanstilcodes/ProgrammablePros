"""Snake Feature Extraction — Genesis 2.0 Track EVA 02 / Phase 2a.

Produces SnakeFeatures dataclass and deterministic compact context string (<= 400 chars).
Computes features ONLY from current state and memory (NO search, NO lookahead, NO BFS/A*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

OPPOSITES: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# Relative heading mapping (ahead, left, right)
RELATIVE_DIRS: dict[str, dict[str, str]] = {
    "up": {"ahead": "up", "left": "left", "right": "right"},
    "down": {"ahead": "down", "left": "right", "right": "left"},
    "left": {"ahead": "left", "left": "down", "right": "up"},
    "right": {"ahead": "right", "left": "up", "right": "down"},
}


@dataclass(frozen=True)
class SnakeFeatures:
    """Rigorous feature representation of the current Snake board state and memory."""

    head: tuple[int, int]
    direction: str
    body_length: int
    board_w: int
    board_h: int

    # Local geometry (NO search)
    free_neighbors: dict[str, bool]
    body_ahead: bool
    body_left: bool
    body_right: bool
    wall_ahead: bool
    wall_left: bool
    wall_right: bool
    distance_to_wall: dict[str, int]

    # Food (only if the API exposes it)
    food: tuple[int, int] | None
    food_delta: tuple[int, int] | None
    food_manhattan: int | None
    food_bearing: str | None
    food_axis_aligned: bool

    # Body topology (local, no search)
    neck: tuple[int, int]
    tail: tuple[int, int]
    tail_manhattan: int
    body_occupies_ahead_2: bool
    occupied_fraction: float

    # Memory-derived
    visit_count_at_head: int
    recent_actions: tuple[str, ...]
    recent_failure_warning: str
    food_stall_turns: int

    # Derived safety flag (FEATURE ONLY, not a decision)
    cramped: bool

    def __getitem__(self, key: str) -> Any:
        """Backward-compatibility mapping for existing test and viewer access."""
        if hasattr(self, key):
            return getattr(self, key)
        if key == "player_pos":
            return self.head
        if key == "goal_pos":
            return self.food
        if key == "distance_to_goal":
            return self.food_manhattan if self.food_manhattan is not None else 0
        if key == "bearing":
            return self.food_bearing if self.food_bearing is not None else "HERE"
        if key == "facing":
            return self.direction
        if key == "obstacles":
            # True means cell is blocked/obstacle
            return {d: not self.free_neighbors.get(d, False) for d in DIRECTIONS}
        if key == "snake_length":
            return self.body_length
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def compute_bearing(dx: int, dy: int) -> str:
    """Calculate 8-point compass bearing from head to food."""
    if dx == 0 and dy == 0:
        return "HERE"
    bearing = ""
    if dy < 0:
        bearing += "N"
    elif dy > 0:
        bearing += "S"
    if dx > 0:
        bearing += "E"
    elif dx < 0:
        bearing += "W"
    return bearing or "HERE"


class FeatureExtractor:
    """Computes SnakeFeatures from GameStateView or state dict and GameMemory."""

    def __init__(
        self,
        default_w: int = 16,
        default_h: int = 16,
        grid_size: int | None = None,
    ) -> None:
        if grid_size is not None:
            self.default_w = grid_size
            self.default_h = grid_size
        else:
            self.default_w = default_w
            self.default_h = default_h

    def extract_features(
        self,
        state_view: Any,
        memory: Any,
    ) -> SnakeFeatures:
        """Extract all required Phase 2a features."""
        # Unpack head
        head_raw = getattr(state_view, "head", None)
        if head_raw is None and isinstance(state_view, dict):
            head_raw = state_view.get("head") or state_view.get("snake", [[0, 0]])[0]
        head: tuple[int, int] = (int(head_raw[0]), int(head_raw[1])) if head_raw else (0, 0)

        # Unpack direction
        dir_raw = getattr(state_view, "direction", None)
        if dir_raw is None and isinstance(state_view, dict):
            dir_raw = state_view.get("direction", "right")
        direction = str(dir_raw).lower() if dir_raw in DIRECTIONS else "right"

        # Unpack body
        body_raw = getattr(state_view, "body", None)
        if body_raw is None and isinstance(state_view, dict):
            body_raw = state_view.get("snake") or state_view.get("body")
        if body_raw:
            body: list[tuple[int, int]] = [(int(p[0]), int(p[1])) for p in body_raw]
        else:
            body = [head]
        body_length = len(body)

        # Board dimensions
        board_w = getattr(state_view, "width", None) or (state_view.get("width") if isinstance(state_view, dict) else None) or self.default_w
        board_h = getattr(state_view, "height", None) or (state_view.get("height") if isinstance(state_view, dict) else None) or self.default_h

        # Food
        food_raw = getattr(state_view, "food", None)
        if food_raw is None and isinstance(state_view, dict):
            food_raw = state_view.get("food")
        
        food: tuple[int, int] | None = None
        food_delta: tuple[int, int] | None = None
        food_manhattan: int | None = None
        food_bearing: str | None = None
        food_axis_aligned: bool = False

        if food_raw is not None and isinstance(food_raw, (list, tuple)) and len(food_raw) >= 2:
            food = (int(food_raw[0]), int(food_raw[1]))
            dx = food[0] - head[0]
            dy = food[1] - head[1]
            food_delta = (dx, dy)
            food_manhattan = abs(dx) + abs(dy)
            food_bearing = compute_bearing(dx, dy)
            food_axis_aligned = (dx == 0 or dy == 0)

        # Body sets
        body_set = set(body)
        body_without_tail = set(body[:-1]) if len(body) > 1 else body_set

        # Reversal direction
        reversal = OPPOSITES.get(direction, "")

        # Local geometry
        free_neighbors: dict[str, bool] = {}
        distance_to_wall: dict[str, int] = {
            "up": head[1],
            "down": (board_h - 1) - head[1],
            "left": head[0],
            "right": (board_w - 1) - head[0],
        }

        for d in DIRECTIONS:
            delta = DELTAS[d]
            nx, ny = head[0] + delta[0], head[1] + delta[1]
            in_bounds = (0 <= nx < board_w) and (0 <= ny < board_h)
            not_body = (nx, ny) not in body_set
            # Reversal is excluded from legal free neighbors
            is_free = in_bounds and not_body and (d != reversal)
            free_neighbors[d] = is_free

        # Relative directions
        rel = RELATIVE_DIRS.get(direction, RELATIVE_DIRS["right"])
        ahead_d = rel["ahead"]
        left_d = rel["left"]
        right_d = rel["right"]

        ahead_pos = (head[0] + DELTAS[ahead_d][0], head[1] + DELTAS[ahead_d][1])
        left_pos = (head[0] + DELTAS[left_d][0], head[1] + DELTAS[left_d][1])
        right_pos = (head[0] + DELTAS[right_d][0], head[1] + DELTAS[right_d][1])

        ahead_in_bounds = (0 <= ahead_pos[0] < board_w) and (0 <= ahead_pos[1] < board_h)
        left_in_bounds = (0 <= left_pos[0] < board_w) and (0 <= left_pos[1] < board_h)
        right_in_bounds = (0 <= right_pos[0] < board_w) and (0 <= right_pos[1] < board_h)

        wall_ahead = not ahead_in_bounds
        wall_left = not left_in_bounds
        wall_right = not right_in_bounds

        body_ahead = ahead_pos in body_set if ahead_in_bounds else False
        body_left = left_pos in body_set if left_in_bounds else False
        body_right = right_pos in body_set if right_in_bounds else False

        # Two cells ahead
        ahead_2_pos = (head[0] + 2 * DELTAS[ahead_d][0], head[1] + 2 * DELTAS[ahead_d][1])
        ahead_2_in_bounds = (0 <= ahead_2_pos[0] < board_w) and (0 <= ahead_2_pos[1] < board_h)
        body_occupies_ahead_2 = (ahead_2_pos in body_set) if ahead_2_in_bounds else False

        # Topology
        neck = body[1] if len(body) > 1 else head
        tail = body[-1] if body else head
        tail_manhattan = abs(head[0] - tail[0]) + abs(head[1] - tail[1])
        total_cells = max(1, board_w * board_h)
        occupied_fraction = round(body_length / total_cells, 4)

        # Memory derived
        visit_count_at_head = 0
        recent_actions: tuple[str, ...] = ()
        recent_failure_warning = ""
        food_stall_turns = 0

        if memory is not None:
            if hasattr(memory, "get_visit_count"):
                visit_count_at_head = memory.get_visit_count(head)
            elif hasattr(memory, "visit_counts"):
                visit_count_at_head = memory.visit_counts.get(head, 0)

            if hasattr(memory, "warning_text"):
                recent_failure_warning = memory.warning_text()

            if hasattr(memory, "food_stall_turns"):
                food_stall_turns = memory.food_stall_turns

            if hasattr(memory, "recent_actions"):
                recent_actions = tuple(memory.recent_actions[-3:])
            elif hasattr(memory, "recent_outcomes"):
                recent_actions = tuple(outcome[0] for outcome in list(memory.recent_outcomes)[-3:])

        # Cramped safety flag: free legal neighbors <= 1
        free_count = sum(1 for v in free_neighbors.values() if v)
        cramped = free_count <= 1

        return SnakeFeatures(
            head=head,
            direction=direction,
            body_length=body_length,
            board_w=board_w,
            board_h=board_h,
            free_neighbors=free_neighbors,
            body_ahead=body_ahead,
            body_left=body_left,
            body_right=body_right,
            wall_ahead=wall_ahead,
            wall_left=wall_left,
            wall_right=wall_right,
            distance_to_wall=distance_to_wall,
            food=food,
            food_delta=food_delta,
            food_manhattan=food_manhattan,
            food_bearing=food_bearing,
            food_axis_aligned=food_axis_aligned,
            neck=neck,
            tail=tail,
            tail_manhattan=tail_manhattan,
            body_occupies_ahead_2=body_occupies_ahead_2,
            occupied_fraction=occupied_fraction,
            visit_count_at_head=visit_count_at_head,
            recent_actions=recent_actions,
            recent_failure_warning=recent_failure_warning,
            food_stall_turns=food_stall_turns,
            cramped=cramped,
        )

    def extract(self, view: Any, memory: Any) -> SnakeFeatures:
        return self.extract_features(view, memory)

    def to_context_string(self, features: SnakeFeatures, subgoal_name: str = "CHASE_FOOD") -> str:
        return format_context_string(features, subgoal_name)


def format_context_string(features: SnakeFeatures, subgoal_name: str = "CHASE_FOOD") -> str:
    """Format features and active subgoal into deterministic string (target <= 320, max 400 chars)."""
    tokens = [
        "game=snake",
        f"head=({features.head[0]},{features.head[1]})",
        f"dir={features.direction}",
        f"len={features.body_length}",
    ]

    if features.food is not None and features.food_delta is not None:
        tokens.extend([
            f"food=({features.food[0]},{features.food[1]})",
            f"fdelta=({features.food_delta[0]},{features.food_delta[1]})",
            f"fdist={features.food_manhattan}",
            f"fbearing={features.food_bearing or 'HERE'}",
        ])
    else:
        tokens.append("food=UNKNOWN")

    u_bit = 1 if features.free_neighbors.get("up", False) else 0
    d_bit = 1 if features.free_neighbors.get("down", False) else 0
    l_bit = 1 if features.free_neighbors.get("left", False) else 0
    r_bit = 1 if features.free_neighbors.get("right", False) else 0
    tokens.append(f"free=[U{u_bit},D{d_bit},L{l_bit},R{r_bit}]")

    wall_dist = features.distance_to_wall.get(features.direction, 0)
    body_ahead_bit = 1 if features.body_ahead else 0
    tokens.extend([
        f"wall_ahead={wall_dist}",
        f"body_ahead={body_ahead_bit}",
        f"tail_dist={features.tail_manhattan}",
        f"occupied={features.occupied_fraction:.2f}",
        f"stall={features.food_stall_turns}",
        f"visited_head={features.visit_count_at_head}",
        f"subgoal={subgoal_name}",
    ])

    if features.recent_failure_warning:
        tokens.append(features.recent_failure_warning)

    ctx = " ".join(tokens)
    if len(ctx) > 400:
        ctx = ctx[:400]
    return ctx


# Alias for backward compatibility
SnakeFeatureExtractor = FeatureExtractor
