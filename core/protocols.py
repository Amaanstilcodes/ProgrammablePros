"""Core Protocols — Genesis 2.0 Track EVA 02.

Defines the structural typing protocols for all modular components in the
decision pipeline. Ensures zero tight coupling between games and core engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class StateParser(Protocol):
    """Parses raw JSON from game server into a structured, validated state representation."""

    def parse(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        """Validate and extract structured fields from raw server state."""
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extracts high-signal deterministic feature vector from parsed state and memory."""

    def extract_features(
        self,
        parsed_state: dict[str, Any],
        memory: Any,
    ) -> dict[str, Any]:
        """Compute egocentric / tactical features for hypothesis generation."""
        ...


@runtime_checkable
class SubgoalArbiter(Protocol):
    """Selects tactical subgoal (e.g. FOOD, MERGE, CORNER) based on current state."""

    def arbitrate(self, state: dict[str, Any], memory: Any) -> Any:
        """Return the active tactical subgoal."""
        ...


@runtime_checkable
class HypothesisBuilder(Protocol):
    """Formulates typed binary hypotheses evaluated by Laya."""

    def build_hypotheses(
        self,
        state: dict[str, Any],
        subgoal: Any,
        legal_actions: list[str],
    ) -> dict[str, Any]:
        """Construct typed questions for legality, progress, danger, and novelty."""
        ...


@runtime_checkable
class GameMemory(Protocol):
    """Tracks spatial and action outcome history across turns."""

    def observe(
        self,
        before_state: dict[str, Any],
        action: str,
        after_state: dict[str, Any],
    ) -> None:
        """Update internal memory representations."""
        ...


@runtime_checkable
class GameClient(Protocol):
    """HTTP client interface for TopGameAI Arena endpoints."""

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        """Start a new game session and return (session_id, initial_state)."""
        ...

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> Any:
        """Send a move to the game server and return structured MoveResult."""
        ...

    def close(self) -> None:
        """Close underlying network session."""
        ...


@runtime_checkable
class LayaBackend(Protocol):
    """Inference backend for the Laya calibrated probability model."""

    def predict_probability(self, context: str, question: str) -> float:
        """Evaluate a typed hypothesis question against context, returning [0.0, 1.0]."""
        ...

    def batch_predict(
        self,
        context: str,
        questions: list[str],
    ) -> list[float]:
        """Batch evaluate hypothesis questions against context."""
        ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """Policy orchestrating EV synthesis and action selection from Laya probabilities."""

    def select_action(
        self,
        action_probabilities: dict[str, float],
        ev_scores: dict[str, float],
        legal_actions: list[str],
        state_before: dict[str, Any],
    ) -> tuple[str, str]:
        """Return (chosen_action, source) where source is 'laya' or 'fallback'."""
        ...
