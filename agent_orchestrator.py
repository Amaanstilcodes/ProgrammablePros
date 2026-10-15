"""Agent Orchestrator Module.

Coordinates the end-to-end decision-layer pipeline:
1. Observe (WorldMemory spatial update)
2. Compress (Egocentric POV transformation)
3. Subgoal (Deterministic hierarchical goal arbitration)
4. Hypothesize (4-way typed question formulation)
5. Laya Inference (Calibrated probability estimation)
6. Calibrate (Platt scaling post-processing)
7. Decide (Multi-objective Expected Value argmax)
8. Log (Append-only JSONL streaming telemetry)
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from advanced_ev_calculator import AdvancedEVCalculator
from calibration_engine import CalibrationEngine
from decision_logger import DecisionLogger
from egocentric_compressor import EgocentricCompressor
from hypothesis_engine import HypothesisEngine
from subgoal_manager import Subgoal, SubgoalManager
from world_memory import WorldMemory


class AgentOrchestrator:
    """Master orchestrator integrating perception, reasoning, calibration, and decision EV."""

    def __init__(self, log_filepath: str = "decision_log.jsonl") -> None:
        """Instantiate all core pipeline subsystems."""
        self.memory: WorldMemory = WorldMemory()
        self.compressor: EgocentricCompressor = EgocentricCompressor()
        self.subgoal_manager: SubgoalManager = SubgoalManager()
        self.hypothesis_engine: HypothesisEngine = HypothesisEngine()
        self.calibrator: CalibrationEngine = CalibrationEngine()
        self.ev_calculator: AdvancedEVCalculator = AdvancedEVCalculator()
        self.logger: DecisionLogger = DecisionLogger(log_filepath=log_filepath)

    @staticmethod
    def _normalize_pos(pos: Any) -> Tuple[int, int]:
        """Convert position representation to (x, y) tuple."""
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
        if isinstance(pos, (int, float)):
            return int(pos), 0
        return 0, 0

    def evaluate_outcome(
        self,
        previous_state: Optional[Dict[str, Any]],
        current_state: Dict[str, Any],
        chosen_action: Optional[str],
    ) -> str:
        """Evaluate transition outcome between previous and current state.

        Returns:
            Status string: 'INITIAL_STATE', 'GOAL_REACHED', 'DAMAGE_TAKEN',
                           'FAILED_NO_PROGRESS', or 'SUCCESS'.
        """
        if previous_state is None or chosen_action is None:
            return "INITIAL_STATE"

        prev_pos = self._normalize_pos(previous_state.get("player_pos", (0, 0)))
        curr_pos = self._normalize_pos(current_state.get("player_pos", prev_pos))

        prev_health = previous_state.get("health", 100)
        curr_health = current_state.get("health", prev_health)

        if current_state.get("status") == "WON":
            return "GOAL_REACHED"

        if curr_health < prev_health:
            return "DAMAGE_TAKEN"

        if current_state.get("hidden_wall_discovered"):
            return "HIDDEN_WALL_COLLISION"

        if curr_pos == prev_pos:
            return "FAILED_NO_PROGRESS"

        return "SUCCESS"

    def _mock_laya_inference(
        self,
        prompts: List[str],
        legal_actions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate hypothesis queries and return calibrated likelihoods.

        Args:
            prompts: Flat list of formatted hypothesis prompt strings.
            legal_actions: Optional list of action strings to structure output.

        Returns:
            Nested dictionary: {action: {question_type: probability_float}}.
        """
        # TODO: Replace with actual client.predict_probability() on hackathon day
        results: Dict[str, Dict[str, float]] = {}

        action_pattern = re.compile(r"action '(\w+)'", re.IGNORECASE)

        for prompt in prompts:
            # Parse action name from prompt text
            match = action_pattern.search(prompt)
            if not match:
                continue
            act = match.group(1).lower()

            if act not in results:
                results[act] = {}

            # Identify hypothesis category from query wording
            if "legal move" in prompt:
                qtype = "is_legal"
                prob = round(random.uniform(0.70, 0.95), 3)
            elif "reduce the distance" in prompt:
                qtype = "toward_subgoal"
                prob = round(random.uniform(0.35, 0.85), 3)
            elif "immediate danger" in prompt or "HAZARD" in prompt:
                qtype = "is_dangerous"
                prob = round(random.uniform(0.05, 0.30), 3)
            elif "UNKNOWN or unvisited" in prompt:
                qtype = "is_novel"
                prob = round(random.uniform(0.20, 0.70), 3)
            else:
                qtype = "general"
                prob = round(random.uniform(0.10, 0.90), 3)

            results[act][qtype] = prob

        # Ensure all legal actions have a complete 4-tuple of probabilities
        if legal_actions:
            for act in legal_actions:
                act_clean = act.lower()
                if act_clean not in results:
                    results[act_clean] = {}
                for default_q in ["is_legal", "toward_subgoal", "is_dangerous", "is_novel"]:
                    if default_q not in results[act_clean]:
                        results[act_clean][default_q] = round(random.uniform(0.20, 0.80), 3)

        return results

    def play_turn(
        self,
        session_id: str,
        turn_number: int,
        current_state: Dict[str, Any],
        legal_actions: List[str],
        previous_state: Optional[Dict[str, Any]] = None,
        previous_action: Optional[str] = None,
    ) -> str:
        """Execute the complete sequential perception-reasoning-decision pipeline.

        Pipeline steps:
            1. Observe: Update WorldMemory with outcome of previous action.
            2. Compress: Generate egocentric POV context string.
            3. Subgoal: Arbitrate tactical objective (RETREAT -> KEY -> EXIT -> EXPLORE).
            4. Hypothesize: Generate 4 typed scientific questions per action + anchor probes.
            5. Laya Inference: Obtain probability estimates (mock/live).
            6. Calibrate: Apply Platt scaling to raw probabilities.
            7. Decide: Multi-objective Expected Value argmax action selection.
            8. Log: Write complete intermediate telemetry to decision_log.jsonl.
            9. Return: Selected best action string.

        Args:
            session_id: Active game session ID.
            turn_number: Monotonic 1-indexed step count.
            current_state: Current game environment state dictionary.
            legal_actions: Available legal actions for this turn.
            previous_state: State dictionary prior to previous action (if any).
            previous_action: Action dispatched on previous turn (if any).

        Returns:
            Chosen action string (e.g., 'up', 'down', 'left', 'right').
        """
        # Step 1: Observe previous transition in spatial memory
        if previous_state is not None and previous_action is not None:
            self.memory.observe(previous_state, previous_action, current_state)

        outcome_desc = self.evaluate_outcome(previous_state, current_state, previous_action)

        # Step 2: Compress state into egocentric POV representation
        compressed_context = self.compressor.compress(current_state, self.memory)

        # Step 3: Hierarchical subgoal arbitration
        subgoal: Subgoal = self.subgoal_manager.get_current_subgoal(current_state, self.memory)

        # Step 4: Formulate typed hypothesis queries
        hypotheses_dict = self.hypothesis_engine.generate_hypotheses(
            compressed_context=compressed_context,
            legal_actions=legal_actions,
            subgoal=subgoal,
            memory=self.memory,
        )
        flattened_queries = self.hypothesis_engine.flatten_queries(hypotheses_dict)
        exact_prompts = [q[2] for q in flattened_queries]

        # Step 5: Laya probability inference (Mock / Live API toggle)
        raw_laya_probs = self._mock_laya_inference(exact_prompts, legal_actions=legal_actions)

        # Step 6: Post-hoc probability calibration (Platt scaling)
        calibrated_probs: Dict[str, Dict[str, float]] = {}
        for act, qmap in raw_laya_probs.items():
            calibrated_probs[act] = {}
            for qtype, p_val in qmap.items():
                calibrated_probs[act][qtype] = round(self.calibrator.predict(p_val), 4)

        # Step 7: Multi-objective Expected Value optimization
        current_pos = current_state.get("player_pos", (0, 0))
        recent_failures = self.memory.get_recent_failures(current_pos)

        chosen_action, ev_breakdown = self.ev_calculator.calculate(
            current_pos=current_pos,
            legal_actions=legal_actions,
            probs=calibrated_probs,
            recent_failures=recent_failures,
        )

        # Step 8: Log decision and telemetry to JSONL
        self.logger.append_decision(
            session_id=session_id,
            turn_number=turn_number,
            raw_state=current_state,
            compressed_context=compressed_context,
            subgoal=subgoal,
            exact_prompts_sent=exact_prompts,
            raw_laya_probs=raw_laya_probs,
            calibrated_probs=calibrated_probs,
            ev_breakdown=ev_breakdown,
            chosen_action=chosen_action,
            next_state=current_state,  # Updated after game client step execution
            outcome=outcome_desc,
        )

        # Step 9: Return chosen action
        return chosen_action


if __name__ == "__main__":
    import os

    test_log = "test_orchestrator_log.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    orchestrator = AgentOrchestrator(log_filepath=test_log)
    sess_id = "test-session-agent-v2"

    print("==================================================================")
    print("  AgentOrchestrator -- 3-Turn End-to-End Simulation")
    print("==================================================================")

    # Turn 1: Initial state at (0, 0)
    state_turn1 = {
        "player_pos": (0, 0),
        "facing": "RIGHT",
        "key_pos": (2, 2),
        "keys_held": 0,
        "goal_pos": (4, 4),
        "health": 100,
        "ammo": 3,
        "steps_remaining": 30,
    }
    legal_1 = ["up", "down", "left", "right"]
    action_1 = orchestrator.play_turn(
        session_id=sess_id,
        turn_number=1,
        current_state=state_turn1,
        legal_actions=legal_1,
        previous_state=None,
        previous_action=None,
    )
    print(f"[Turn 1] Pos: (0,0) -> Chosen Action: '{action_1.upper()}'")

    # Turn 2: Move successfully to (1, 0)
    state_turn2 = {
        "player_pos": (1, 0),
        "facing": "RIGHT",
        "key_pos": (2, 2),
        "keys_held": 0,
        "goal_pos": (4, 4),
        "health": 100,
        "ammo": 3,
        "steps_remaining": 29,
    }
    legal_2 = ["up", "down", "left", "right"]
    action_2 = orchestrator.play_turn(
        session_id=sess_id,
        turn_number=2,
        current_state=state_turn2,
        legal_actions=legal_2,
        previous_state=state_turn1,
        previous_action=action_1,
    )
    print(f"[Turn 2] Pos: (1,0) -> Chosen Action: '{action_2.upper()}'")

    # Turn 3: Action failed / hit wall at (1, -1) -> stay at (1, 0)
    state_turn3 = {
        "player_pos": (1, 0),
        "facing": "RIGHT",
        "key_pos": (2, 2),
        "keys_held": 0,
        "goal_pos": (4, 4),
        "health": 100,
        "ammo": 3,
        "steps_remaining": 28,
    }
    legal_3 = ["up", "down", "left", "right"]
    action_3 = orchestrator.play_turn(
        session_id=sess_id,
        turn_number=3,
        current_state=state_turn3,
        legal_actions=legal_3,
        previous_state=state_turn2,
        previous_action="up",  # Intentionally simulate failed 'up' move
    )
    print(f"[Turn 3] Pos: (1,0) -> Chosen Action: '{action_3.upper()}'")

    # Verify log output
    logs = orchestrator.logger.read_recent_logs()
    assert len(logs) == 3, f"Expected 3 logs, got {len(logs)}"
    assert logs[0]["turn_number"] == 1
    assert logs[1]["turn_number"] == 2
    assert logs[2]["turn_number"] == 3
    assert logs[2]["outcome"] == "FAILED_NO_PROGRESS"

    print(f"\n[PASS] Verified {len(logs)} decisions logged to '{test_log}'.")
    os.remove(test_log)
    print("==================================================================")
