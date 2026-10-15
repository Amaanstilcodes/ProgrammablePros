"""Subgoal Manager Module.

Implements deterministic hierarchical goal arbitration for the decision layer.
Dynamically shifts tactical focus between survival (RETREAT), prerequisite
acquisition (KEY), mission completion (EXIT), and frontier mapping (EXPLORE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from world_memory import WorldMemory, STATUS_FREE, STATUS_WALL, STATUS_UNKNOWN


@dataclass
class Subgoal:
    """Representation of the agent's current high-level tactical objective.

    Attributes:
        target_type: Subgoal category ('RETREAT', 'KEY', 'EXIT', 'EXPLORE').
        coordinates: Target (x, y) grid coordinates, or None if non-spatial.
        description: Natural-language rationale explaining why this target was selected.
    """
    target_type: str
    coordinates: Optional[Tuple[int, int]]
    description: str


class SubgoalManager:
    """Arbitrates subgoals using a strict, deterministic priority cascade."""

    def __init__(self, retreat_health_threshold: int = 20, threat_distance_threshold: int = 3) -> None:
        """Initialize SubgoalManager with configurable risk thresholds.

        Args:
            retreat_health_threshold: Health level below which retreat takes top priority (default: 20).
            threat_distance_threshold: Manhattan distance within which an enemy triggers retreat (default: 3).
        """
        self.retreat_health_threshold: int = retreat_health_threshold
        self.threat_distance_threshold: int = threat_distance_threshold

    @staticmethod
    def _normalize_pos(pos: Any) -> Optional[Tuple[int, int]]:
        """Safely convert coordinate representations to (x, y) tuples."""
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
        if isinstance(pos, (int, float)):
            return int(pos), 0
        return None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_retreat_cell(
        self,
        player_pos: Tuple[int, int],
        threat_pos: Tuple[int, int],
        memory: WorldMemory,
    ) -> Tuple[int, int]:
        """Identify a known safe cell that maximizes distance from the threat."""
        best_cell = player_pos
        max_threat_dist = self._manhattan(player_pos, threat_pos)

        # Evaluate all known FREE cells in memory
        for coord, status in memory.cells.items():
            if status == STATUS_FREE:
                threat_dist = self._manhattan(coord, threat_pos)
                player_dist = self._manhattan(player_pos, coord)
                # Prioritize cells further from threat, keeping player transit distance bounded
                if threat_dist > max_threat_dist and player_dist <= 3:
                    max_threat_dist = threat_dist
                    best_cell = coord

        # Fallback: step directly away from threat along primary vector
        if best_cell == player_pos:
            dx = player_pos[0] - threat_pos[0]
            dy = player_pos[1] - threat_pos[1]
            step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
            step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
            candidate = (player_pos[0] + step_x, player_pos[1] + step_y)
            if memory.get_cell(candidate[0], candidate[1]) != STATUS_WALL:
                best_cell = candidate

        return best_cell

    def _find_unexplored_frontier(
        self,
        player_pos: Tuple[int, int],
        memory: WorldMemory,
    ) -> Tuple[int, int]:
        """Find the nearest unexplored frontier cell adjacent to known free space."""
        # 1. Search immediate 4-neighborhood of player for UNKNOWN cells
        candidates: List[Tuple[int, int]] = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (player_pos[0] + dx, player_pos[1] + dy)
            if memory.get_cell(neighbor[0], neighbor[1]) == STATUS_UNKNOWN:
                candidates.append(neighbor)

        if candidates:
            # Sort by least visits of surrounding space
            return candidates[0]

        # 2. Check all known UNKNOWN coordinates recorded in memory
        unknown_cells = [
            coord for coord, status in memory.cells.items()
            if status == STATUS_UNKNOWN
        ]
        if unknown_cells:
            unknown_cells.sort(key=lambda c: self._manhattan(player_pos, c))
            return unknown_cells[0]

        # 3. Fallback: least visited adjacent FREE cell
        least_visited_cell = player_pos
        min_visits = float("inf")
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (player_pos[0] + dx, player_pos[1] + dy)
            if memory.get_cell(neighbor[0], neighbor[1]) != STATUS_WALL:
                visits = memory.get_visit_count(neighbor[0], neighbor[1])
                if visits < min_visits:
                    min_visits = visits
                    least_visited_cell = neighbor

        return least_visited_cell

    def select_subgoal(self, state: Dict[str, Any], memory: WorldMemory) -> Subgoal:
        """Arbitrate and select the active subgoal using strict deterministic priority.

        Priorities:
            1. RETREAT: Low health (<20) with nearby threat (<3 dist).
            2. KEY: Need key to unlock exit (keys_held == 0).
            3. EXIT: Key held, proceed to goal/exit.
            4. EXPLORE: Fallback to nearest unexplored frontier.

        Args:
            state: Current game environment state dictionary.
            memory: WorldMemory instance with spatial knowledge and visit counts.

        Returns:
            Selected Subgoal dataclass instance.
        """
        player_pos = self._normalize_pos(state.get("player_pos", (0, 0))) or (0, 0)
        health = state.get("health")
        enemy_pos = self._normalize_pos(state.get("guard_pos") or state.get("enemy_pos"))
        keys_held = state.get("keys_held", state.get("keys", 0))
        key_pos = self._normalize_pos(state.get("key_pos"))
        goal_pos = self._normalize_pos(state.get("goal_pos") or state.get("exit_pos"))

        # Calculate enemy distance if present
        enemy_dist = self._manhattan(player_pos, enemy_pos) if enemy_pos and enemy_pos[0] >= 0 else 999

        # ──────────────────────────────────────────────────────────
        # Priority 1: Survival / Retreat
        # ──────────────────────────────────────────────────────────
        if health is not None and health < self.retreat_health_threshold and enemy_dist < self.threat_distance_threshold:
            retreat_coords = self._find_retreat_cell(player_pos, enemy_pos, memory)  # type: ignore[arg-type]
            return Subgoal(
                target_type="RETREAT",
                coordinates=retreat_coords,
                description=f"Health critical ({health}<{self.retreat_health_threshold}) and threat within range ({enemy_dist}). Find cover.",
            )

        # ──────────────────────────────────────────────────────────
        # Priority 2: Key Acquisition
        # ──────────────────────────────────────────────────────────
        # Triggered if keys_held == 0 and key location is known or door is locked
        if keys_held == 0 and (key_pos is not None or "DOOR" in memory.cells.values()):
            return Subgoal(
                target_type="KEY",
                coordinates=key_pos,
                description="Acquire key to unlock the exit.",
            )

        # ──────────────────────────────────────────────────────────
        # Priority 3: Mission Completion / Exit
        # ──────────────────────────────────────────────────────────
        if (keys_held > 0 or key_pos is None) and goal_pos is not None:
            return Subgoal(
                target_type="EXIT",
                coordinates=goal_pos,
                description="Key secured (or unneeded). Proceed directly to the exit.",
            )

        # ──────────────────────────────────────────────────────────
        # Priority 4: Frontier Exploration
        # ──────────────────────────────────────────────────────────
        frontier_coords = self._find_unexplored_frontier(player_pos, memory)
        return Subgoal(
            target_type="EXPLORE",
            coordinates=frontier_coords,
            description="Explore unknown territory to reveal path and objectives.",
        )

    def get_current_subgoal(self, state: Dict[str, Any], memory: WorldMemory) -> Subgoal:
        """Alias for select_subgoal to match Agent V2 orchestration specification."""
        return self.select_subgoal(state, memory)



if __name__ == "__main__":
    # Self-test demonstration
    mem = WorldMemory()
    manager = SubgoalManager()

    # Test 1: Priority 2 (KEY acquisition)
    state_key = {"player_pos": (1, 1), "keys_held": 0, "key_pos": (3, 3), "goal_pos": (7, 7)}
    sg1 = manager.select_subgoal(state_key, mem)
    assert sg1.target_type == "KEY"
    assert sg1.coordinates == (3, 3)

    # Test 2: Priority 3 (EXIT after key secured)
    state_exit = {"player_pos": (3, 3), "keys_held": 1, "key_pos": (3, 3), "goal_pos": (7, 7)}
    sg2 = manager.select_subgoal(state_exit, mem)
    assert sg2.target_type == "EXIT"
    assert sg2.coordinates == (7, 7)

    # Test 3: Priority 1 (RETREAT on low health)
    state_retreat = {
        "player_pos": (2, 2),
        "health": 15,
        "enemy_pos": (2, 3),
        "keys_held": 0,
        "key_pos": (5, 5),
    }
    sg3 = manager.select_subgoal(state_retreat, mem)
    assert sg3.target_type == "RETREAT"

    # Test 4: Priority 4 (EXPLORE fallback)
    state_explore = {"player_pos": (0, 0), "keys_held": 0}
    sg4 = manager.select_subgoal(state_explore, mem)
    assert sg4.target_type == "EXPLORE"

    print(f"[SubgoalManager self-test PASSED]")
    print(f"  Key Subgoal:     {sg1}")
    print(f"  Exit Subgoal:    {sg2}")
    print(f"  Retreat Subgoal: {sg3}")
    print(f"  Explore Subgoal: {sg4}")
