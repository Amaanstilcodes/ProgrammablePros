"""Hypothesis Engine Module.

Generates 4-dimensional typed binary hypothesis questions per legal action:
- is_legal: Valid transition to a non-wall cell
- toward_subgoal: Reduces metric distance to active tactical subgoal
- is_dangerous: Incurs hazard, damage, or hostile enemy exposure
- is_novel: Traverses unvisited or unexplored territory (exploration bonus)

Includes Anchor Probe injection with ground-truth known answers for real-time
calibration drift detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from subgoal_manager import Subgoal
from world_memory import WorldMemory


@dataclass
class ActionHypotheses:
    """Four typed scientific hypothesis questions formulated for a single action."""
    action: str
    is_legal: str
    toward_subgoal: str
    is_dangerous: str
    is_novel: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary keyed by hypothesis category."""
        return {
            "is_legal": self.is_legal,
            "toward_subgoal": self.toward_subgoal,
            "is_dangerous": self.is_dangerous,
            "is_novel": self.is_novel,
        }


@dataclass
class AnchorProbe:
    """Ground-truth calibration probe query for monitoring model drift."""
    id: str
    question: str
    ground_truth: float
    tolerance: float = 0.15


class HypothesisEngine:
    """Formulates typed 4-way hypothesis queries and calibration sanity probes."""

    def __init__(self) -> None:
        """Initialize hypothesis engine with standard ground-truth anchor probes."""
        self._anchor_probes: List[AnchorProbe] = [
            AnchorProbe(
                id="probe_wall_illegal",
                question="Hypothesis: Moving into a cell explicitly marked as WALL in the context will result in a legal move. Probability?",
                ground_truth=0.0,
                tolerance=0.10,
            ),
            AnchorProbe(
                id="probe_math_valid",
                question="Hypothesis: 0.5 is greater than 0.1. Probability?",
                ground_truth=1.0,
                tolerance=0.10,
            ),
        ]

    def generate_hypotheses(
        self,
        compressed_context: str,
        legal_actions: List[str],
        subgoal: Subgoal,
        memory: Optional[WorldMemory] = None,
    ) -> Dict[str, Any]:
        """Generate 4 typed hypothesis questions per legal action and append anchor probes.

        Args:
            compressed_context: Egocentric structured POV string from EgocentricCompressor.
            legal_actions: List of valid action strings (e.g., ['up', 'down', 'left', 'right']).
            subgoal: Current active Subgoal from SubgoalManager.
            memory: Optional WorldMemory instance for contextual enrichment.

        Returns:
            Dictionary with:
                - 'actions': Dict[str, Dict[str, str]] mapping action -> 4 hypothesis queries
                - 'anchor_probes': List[Dict[str, Any]] containing calibration check probes
        """
        action_hypotheses: Dict[str, Dict[str, str]] = {}

        # Strip multi-line context into single clean prompt block
        cleaned_context = " ".join(compressed_context.splitlines())

        for act in legal_actions:
            act_clean = act.strip().lower()

            # Q1: Legality
            q1 = (
                f"Context: {cleaned_context}. "
                f"Hypothesis: Executing action '{act_clean}' will result in a legal move to a FREE cell. "
                f"What is the calibrated probability (0.0 to 1.0) that this is true?"
            )

            # Q2: Subgoal Progress
            q2 = (
                f"Context: {cleaned_context}. "
                f"Hypothesis: Executing action '{act_clean}' will reduce the distance to the current subgoal "
                f"({subgoal.target_type}). What is the calibrated probability (0.0 to 1.0) that this is true?"
            )

            # Q3: Safety / Hazard
            q3 = (
                f"Context: {cleaned_context}. "
                f"Hypothesis: Executing action '{act_clean}' will expose the agent to immediate danger, "
                f"damage, or a HAZARD. What is the calibrated probability (0.0 to 1.0) that this is true?"
            )

            # Q4: Novelty / Exploration
            q4 = (
                f"Context: {cleaned_context}. "
                f"Hypothesis: Executing action '{act_clean}' will move the agent into an UNKNOWN or unvisited cell. "
                f"What is the calibrated probability (0.0 to 1.0) that this is true?"
            )

            hypo = ActionHypotheses(
                action=act_clean,
                is_legal=q1,
                toward_subgoal=q2,
                is_dangerous=q3,
                is_novel=q4,
            )
            action_hypotheses[act_clean] = hypo.to_dict()

        # Build anchor probes payload
        probes_payload = [asdict(p) for p in self._anchor_probes]

        return {
            "actions": action_hypotheses,
            "anchor_probes": probes_payload,
        }

    def inject_anchor_probes(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append ground-truth calibration anchor probes to an arbitrary query list.

        Args:
            hypotheses: Existing list of query dictionaries.

        Returns:
            New list containing all original queries plus ground-truth anchor probes.
        """
        probes = [asdict(p) for p in self._anchor_probes]
        return list(hypotheses) + probes

    def flatten_queries(
        self,
        hypotheses_dict: Dict[str, Any],
    ) -> List[Tuple[str, str, str]]:
        """Flatten structured action hypotheses into a batch-query list for model inference.

        Returns:
            List of tuples: (action, question_type, query_string).
        """
        flat: List[Tuple[str, str, str]] = []
        for action, qmap in hypotheses_dict.get("actions", {}).items():
            for qtype, query_str in qmap.items():
                flat.append((action, qtype, query_str))
        return flat


if __name__ == "__main__":
    from egocentric_compressor import EgocentricCompressor
    from world_memory import WorldMemory

    # Self-test demonstration
    mem = WorldMemory()
    comp = EgocentricCompressor()
    sg_mgr = Subgoal() if False else None

    state = {
        "player_pos": (2, 2),
        "facing": "UP",
        "key_pos": (4, 4),
        "keys_held": 0,
        "goal_pos": (7, 7),
        "health": 100,
        "ammo": 3,
        "steps_remaining": 30,
    }
    context = comp.compress(state, mem)
    subgoal = Subgoal(target_type="KEY", coordinates=(4, 4), description="Acquire key")

    engine = HypothesisEngine()
    result = engine.generate_hypotheses(
        compressed_context=context,
        legal_actions=["up", "down", "left", "right"],
        subgoal=subgoal,
        memory=mem,
    )

    actions = result["actions"]
    probes = result["anchor_probes"]

    assert len(actions) == 4
    for act in ["up", "down", "left", "right"]:
        assert act in actions
        assert "is_legal" in actions[act]
        assert "toward_subgoal" in actions[act]
        assert "is_dangerous" in actions[act]
        assert "is_novel" in actions[act]

    assert len(probes) == 2
    assert probes[0]["ground_truth"] == 0.0
    assert probes[1]["ground_truth"] == 1.0

    flat_list = engine.flatten_queries(result)
    assert len(flat_list) == 16  # 4 actions * 4 questions

    print("[HypothesisEngine self-test PASSED]")
    print(f"  Actions generated: {list(actions.keys())}")
    print(f"  Total flattened queries: {len(flat_list)}")
    print(f"  Anchor probes: {[p['id'] for p in probes]}")
    print(f"  Sample Question (RIGHT -> toward_subgoal):\n    {actions['right']['toward_subgoal'][:100]}...")
