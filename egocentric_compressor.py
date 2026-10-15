"""Egocentric Compressor Module.

Translates absolute global game state and persistent spatial memory into a dense,
egocentric, point-of-view (POV) text description formatted for small calibrated
models like Laya.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from world_memory import WorldMemory, STATUS_UNKNOWN


class EgocentricCompressor:
    """Transforms global game state and WorldMemory into an egocentric prompt representation.

    Format:
        POV: agent at (x,y), facing [DIR]
        LOCAL: AHEAD: [status], LEFT: [status], RIGHT: [status], BEHIND: [status]
        OBJECTIVES: [List known keys/doors/exits with relative bearing and manhattan distance]
        RESOURCES: health=X, ammo=Y
        HISTORY: [Last 3 failed actions from memory]
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _normalize_pos(pos: Any) -> Tuple[int, int]:
        """Convert scalar (1D) or list/tuple (2D) positions to (x, y)."""
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
        if isinstance(pos, (int, float)):
            return int(pos), 0
        return 0, 0

    @staticmethod
    def calculate_bearing(origin: Tuple[int, int], target: Tuple[int, int]) -> str:
        """Compute 8-point compass bearing from origin (px, py) to target (tx, ty).

        Coordinate system:
            +X = East (Right)
            -X = West (Left)
            +Y = South (Down)
            -Y = North (Up)

        Returns:
            Compass string: 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', or 'HERE'.
        """
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]

        if dx == 0 and dy == 0:
            return "HERE"

        # Cardinal alignments
        if dx == 0:
            return "S" if dy > 0 else "N"
        if dy == 0:
            return "E" if dx > 0 else "W"

        # Intercardinal alignments
        if dy < 0 and dx > 0:
            return "NE"
        if dy < 0 and dx < 0:
            return "NW"
        if dy > 0 and dx > 0:
            return "SE"
        return "SW"

    @staticmethod
    def calculate_manhattan(origin: Tuple[int, int], target: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(origin[0] - target[0]) + abs(origin[1] - target[1])

    def _format_objectives(
        self,
        state: Dict[str, Any],
        player_pos: Tuple[int, int],
    ) -> str:
        """Extract and format all known game targets with bearing and distance."""
        objectives: List[str] = []

        # Key target
        key_pos = state.get("key_pos")
        keys_held = state.get("keys_held", 0)
        if key_pos is not None and keys_held == 0:
            kp = self._normalize_pos(key_pos)
            bearing = self.calculate_bearing(player_pos, kp)
            dist = self.calculate_manhattan(player_pos, kp)
            objectives.append(f"KEY(bearing={bearing}, dist={dist})")

        # Goal / Exit target
        goal_pos = state.get("goal_pos") or state.get("exit_pos")
        if goal_pos is not None:
            gp = self._normalize_pos(goal_pos)
            bearing = self.calculate_bearing(player_pos, gp)
            dist = self.calculate_manhattan(player_pos, gp)
            status_label = "EXIT" if "key_pos" in state else "GOAL"
            lock_label = " [LOCKED-NEEDS-KEY]" if (keys_held == 0 and "key_pos" in state) else ""
            objectives.append(f"{status_label}(bearing={bearing}, dist={dist}{lock_label})")

        # Guard / Enemy target
        guard_pos = state.get("guard_pos") or state.get("enemy_pos")
        if guard_pos is not None:
            gp = self._normalize_pos(guard_pos)
            if gp[0] >= 0 and gp[1] >= 0:
                bearing = self.calculate_bearing(player_pos, gp)
                dist = self.calculate_manhattan(player_pos, gp)
                guard_dir = state.get("guard_direction", "NONE")
                objectives.append(f"GUARD(bearing={bearing}, dist={dist}, patrol={guard_dir})")

        # Fallback if no specific positions parsed
        if not objectives:
            return "NONE"

        return ", ".join(objectives)

    def _format_resources(self, state: Dict[str, Any]) -> str:
        """Extract resource metrics: health, ammo, keys, steps remaining."""
        health = state.get("health", 100)
        ammo = state.get("ammo", 0)
        keys = state.get("keys_held", state.get("keys", 0))
        steps = state.get("steps_remaining", state.get("steps_left"))

        parts = [f"health={health}", f"ammo={ammo}"]
        if keys is not None:
            parts.append(f"keys={keys}")
        if steps is not None:
            parts.append(f"steps_left={steps}")

        return ", ".join(parts)

    def _format_history(self, memory: WorldMemory) -> str:
        """Format the last 3 failed actions from memory."""
        failures = memory.get_recent_failures(limit=3)
        if not failures:
            return "NONE"

        formatted: List[str] = []
        for fail in failures:
            pos = fail.get("pos", (0, 0))
            action = str(fail.get("action", "")).upper()
            reason = fail.get("reason", "failed")
            formatted.append(f"action '{action}' at ({pos[0]},{pos[1]}) failed ({reason})")

        return "; ".join(formatted)

    def compress(self, state: Dict[str, Any], memory: WorldMemory) -> str:
        """Produce the standardized egocentric POV string.

        Args:
            state: Global game state dictionary.
            memory: Active WorldMemory tracking spatial map and failure history.

        Returns:
            Structured multi-line egocentric prompt string.
        """
        player_pos = self._normalize_pos(state.get("player_pos", (0, 0)))
        facing = str(state.get("facing", "right")).strip().upper()

        # Query WorldMemory for egocentric local surroundings
        local_context = memory.get_local_context(player_pos[0], player_pos[1], facing_direction=facing)

        ahead = local_context.get("AHEAD", STATUS_UNKNOWN)
        left = local_context.get("LEFT", STATUS_UNKNOWN)
        right = local_context.get("RIGHT", STATUS_UNKNOWN)
        behind = local_context.get("BEHIND", STATUS_UNKNOWN)

        objectives_str = self._format_objectives(state, player_pos)
        resources_str = self._format_resources(state)
        history_str = self._format_history(memory)

        lines = [
            f"POV: agent at ({player_pos[0]},{player_pos[1]}), facing {facing}",
            f"LOCAL: AHEAD: {ahead}, LEFT: {left}, RIGHT: {right}, BEHIND: {behind}",
            f"OBJECTIVES: {objectives_str}",
            f"RESOURCES: {resources_str}",
            f"HISTORY: {history_str}",
        ]

        return "\n".join(lines)
