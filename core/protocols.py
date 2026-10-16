"""Core Protocols & Transport Types — Genesis 2.0 Track EVA 02.

Defines structural typing protocols and dataclasses for game isolation:
- StateParser, FeatureExtractor, SubgoalArbiter, HypothesisBuilder
- GameMemory, GameClient, LayaBackend, DecisionPolicy
- Schema-validated transport dataclasses: GameStateView, SessionStart, MoveResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class SchemaError(ValueError):
    """Raised when incoming raw API state violates the expected contract schema."""
    pass


class ContractNotDiscoveredError(RuntimeError):
    """Raised when an unverified endpoint path or payload shape is invoked before probing."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Core Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GameStateView:
    """Immutable, validated view of parsed game state. Never mutates the raw dictionary."""

    game: str
    session_id: str
    raw: dict[str, Any]  # read-only reference
    head: tuple[int, int] | None = None
    body: tuple[tuple[int, int], ...] | None = None
    direction: str | None = None
    food: tuple[int, int] | None = None
    width: int | None = None
    height: int | None = None
    board: tuple[tuple[int, ...], ...] | None = None
    score: int = 0
    best_tile: int = 0
    step: int | None = None
    game_over: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if key == "max_tile":
                return self.best_tile
            return val
        if key == "snake":
            return list(self.body) if self.body else []
        if key == "max_tile":
            return self.best_tile
        if key in self.extra:
            return self.extra[key]
        if key in self.raw:
            return self.raw[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default



@dataclass(frozen=True)
class SessionStart:
    """Outcome of starting a game session."""

    session_id: str
    raw_state: dict[str, Any]
    raw_response: dict[str, Any]


@dataclass
class MoveResult:
    """Outcome of a single move sent to the game server."""

    raw_response: dict[str, Any]
    state_after: dict[str, Any]
    result: Literal["applied", "rejected"]
    game_over: bool
    api_ms: int
    http_status: int = 200
    request_id: str | None = None


@dataclass
class FailureRecord:
    """Record of an action resulting in failure or rejection."""

    action: str
    step: int
    reason: str


@dataclass
class Subgoal:
    """Active tactical subgoal."""

    name: str
    goal_phrase: str
    priority: int
    rationale: str


@dataclass
class Hypothesis:
    """Typed binary question for probability calibration."""

    id: str
    action: str
    kind: str
    question_text: str
    expected_anchor: float | None = None


@dataclass
class Decision:
    """Synthesized action decision."""

    action: str
    source: Literal["laya", "fallback"]
    ev_breakdown: dict[str, float]
    normalized_laya_probs: dict[str, float]
    questions: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Structural Protocols
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class StateParser(Protocol):
    """Parses raw JSON from game server into a structured, validated state representation."""

    def parse(self, raw: dict[str, Any]) -> Any:
        ...

    def is_game_over(self, raw: dict[str, Any]) -> bool:
        ...

    def is_rejected(self, before: dict[str, Any], after: dict[str, Any], response: dict[str, Any]) -> bool:
        ...

    def score_of(self, raw: dict[str, Any]) -> int:
        ...

    def validate(self, raw: dict[str, Any]) -> None:
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extracts high-signal deterministic feature vector from parsed state and memory."""

    def extract(self, view: Any, memory: Any) -> Any:
        ...

    def to_context_string(self, features: Any) -> str:
        ...


@runtime_checkable
class SubgoalArbiter(Protocol):
    """Selects tactical subgoal based on features and memory."""

    def select(self, features: Any, memory: Any) -> Subgoal:
        ...


@runtime_checkable
class HypothesisBuilder(Protocol):
    """Formulates typed binary hypotheses evaluated by Laya."""

    def build(self, features: Any, subgoal: Any, memory: Any) -> list[Hypothesis]:
        ...


@runtime_checkable
class GameMemory(Protocol):
    """Tracks spatial and action outcome history across turns."""

    def observe_transition(self, before: Any, action: str, after: Any, result: str) -> None:
        ...

    def recent_failures(self) -> list[FailureRecord]:
        ...

    def warning_text(self) -> str:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...


@runtime_checkable
class GameClient(Protocol):
    """HTTP client interface for TopGameAI Arena endpoints."""

    def start_game(self, agent_name: str = "ProgrammablePros") -> Any:
        ...

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        ...

    def get_state(self, session_id: str) -> dict[str, Any]:
        ...

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class LayaBackend(Protocol):
    """Inference backend for Laya probability model."""

    def score_hypotheses(self, context: str, questions: list[str]) -> list[float]:
        ...

    def model_revision(self) -> str:
        ...

    def predict_probability(self, context: str, question: str) -> float:
        ...

    def batch_predict(self, context: str, questions: list[str]) -> list[float]:
        ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """Policy orchestrating EV synthesis and action selection."""

    def decide(
        self,
        features: Any,
        subgoal: Any,
        hypotheses: Any,
        laya_outputs: Any,
        memory: Any,
    ) -> Decision:
        ...
