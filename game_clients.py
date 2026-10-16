"""game_clients.py — Real TopGameAI Arena API Client — Genesis 2.0 Track EVA 02.

Official TopGameAI Arena API integration for Snake and 2048.
Base URL: https://topgameai.com/api/arena

ENDPOINTS:
  1. POST /sessions                      -> Create Session {"game": "snake"|"2048", "agent_name": "ProgrammablePros"}
  2. POST /sessions/{session_id}/action  -> Submit Move {"action": "up"|"down"|"left"|"right"}
  3. GET  /sessions/{session_id}         -> Query session state
  4. GET  /leaderboard/{game}            -> Leaderboard (read-only verification)
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from core.protocols import GameStateView
except ImportError:
    GameStateView = Any  # type: ignore[misc,assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL: str = "https://topgameai.com/api/arena"
DEFAULT_AGENT_NAME: str = "ProgrammablePros"
DEFAULT_TIMEOUT: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions & Result Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class GameAPIError(RuntimeError):
    """Raised on unexpected HTTP status or transport error."""

    def __init__(self, status_code: int, endpoint: str, body: str) -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        self.body = body
        super().__init__(f"Game API error at {endpoint} [HTTP {status_code}]: {body[:300]}")


class ContractNotDiscoveredError(RuntimeError):
    """Retained for backward compatibility with existing exception imports."""
    pass


@dataclass
class MoveResult:
    """Outcome of a single move sent to the game server.

    Fields:
        state_after: Game-specific state object extracted from response["state"].
        result: "applied" if move was accepted and advanced step count; "rejected" otherwise.
        game_over: Boolean indicating if game has ended (from response["done"]).
        api_ms: HTTP request round-trip latency in milliseconds.
        steps: Current step counter from server (from response["steps"]).
    """
    state_after: dict[str, Any]
    result: Literal["applied", "rejected"]
    game_over: bool
    api_ms: int
    steps: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)
    http_status: int = 200
    won: bool = False
    score: int = 0
    request_id: str | None = None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.raw_response:
            return self.raw_response[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.raw_response.get(key, default)


class SessionStart(tuple):
    """Outcome of starting a game session.

    Inherits from tuple so (session_id, initial_state) unpacking works seamlessly:
        session_id, state = client.start_game()
    While preserving attribute access:
        start = client.start_game()
        sid = start.session_id
        state = start.raw_state
    """
    def __new__(
        cls,
        session_id: str,
        raw_state: dict[str, Any],
        raw_response: dict[str, Any] | None = None,
    ) -> SessionStart:
        inst = super().__new__(cls, (session_id, raw_state))
        inst._session_id = session_id
        inst._raw_state = raw_state
        inst._raw_response = raw_response or {}
        return inst

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._raw_state

    @property
    def raw_response(self) -> dict[str, Any]:
        return self._raw_response


# ─────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_retry_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Create requests.Session configured with exponential backoff on 429/5xx."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _normalize_state(
    game: str,
    raw_state: dict[str, Any],
    top_response: dict[str, Any],
    session_id: str = "",
) -> dict[str, Any]:
    """Ensure state dictionary has extracted server state plus enriched metadata."""
    state = copy.deepcopy(raw_state)
    steps = int(top_response.get("steps", state.get("step", 0)))
    score = int(top_response.get("score", state.get("score", 0)))
    done = bool(top_response.get("done", state.get("game_over", False)))
    won = bool(top_response.get("won", False))
    sid = session_id or str(top_response.get("session_id", ""))

    state.setdefault("step", steps)
    state.setdefault("steps", steps)
    state.setdefault("score", score)
    state.setdefault("game_over", done)
    state.setdefault("done", done)
    state.setdefault("won", won)
    if sid:
        state.setdefault("session_id", sid)

    if game == "2048":
        if "grid" in state and "board" not in state:
            state["board"] = [list(row) for row in state["grid"]]
        if "best" in state:
            state.setdefault("best_tile", state["best"])
            state.setdefault("max_tile", state["best"])

    return state


# ─────────────────────────────────────────────────────────────────────────────
# BaseGameClient
# ─────────────────────────────────────────────────────────────────────────────

class BaseGameClient:
    """HTTP client for TopGameAI Arena API.

    Contract:
      Base URL: https://topgameai.com/api/arena
      POST /sessions
      POST /sessions/{session_id}/action
      GET  /sessions/{session_id}
      GET  /leaderboard/{game}
      Authorization: NONE (no auth header per official documentation)
      Content-Type: application/json
      Retry: on 429/5xx with exponential backoff
      Timeout: 10s
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",  # Accepted for signature compatibility; NO auth header sent per contract
        agent_name: str = DEFAULT_AGENT_NAME,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        url = (base_url or os.getenv("TOPGAMEAI_BASE_URL", "") or DEFAULT_BASE_URL).rstrip("/")
        # Ensure URL points to /api/arena
        if url.endswith("/api"):
            url += "/arena"
        elif not url.endswith("/arena") and "topgameai.com" in url:
            if "/api/arena" not in url:
                url = f"{url}/api/arena"

        self.base_url: str = url
        self.agent_name: str = agent_name or DEFAULT_AGENT_NAME
        self.timeout: float = timeout if timeout > 0 else DEFAULT_TIMEOUT

        self.session: requests.Session = _build_retry_session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ProgrammablePros/1.0 (Track2-EVA02)",
        })
        # IMPORTANT: Official documentation requires NO Authorization header. Do NOT add one.

    def create_session(self, game: str, agent_name: str | None = None) -> dict[str, Any]:
        """POST /sessions to create a new arena game session."""
        target_url = f"{self.base_url}/sessions"
        payload = {
            "game": game,
            "agent_name": agent_name or self.agent_name,
        }
        resp = self.session.post(target_url, json=payload, timeout=self.timeout)

        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)

        return resp.json()

    def submit_action(
        self,
        session_id: str,
        action: str,
    ) -> tuple[int, dict[str, Any], int, str | None]:
        """POST /sessions/{session_id}/action to submit a move.

        Returns:
            (status_code, response_json_or_empty, latency_ms, request_id)
        """
        target_url = f"{self.base_url}/sessions/{session_id}/action"
        payload = {"action": action.strip().lower()}

        t0 = time.perf_counter()
        resp = self.session.post(target_url, json=payload, timeout=self.timeout)
        api_ms = int((time.perf_counter() - t0) * 1000.0)
        request_id = resp.headers.get("x-request-id")

        if resp.status_code == 404:
            raise GameAPIError(404, target_url, f"Session '{session_id}' not found: {resp.text}")

        if resp.status_code == 400:
            # Rejection / invalid action (e.g. 180-degree reversal)
            try:
                err_data = resp.json()
            except Exception:
                err_data = {"error": resp.text}
            return 400, err_data, api_ms, request_id

        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)

        try:
            body = resp.json()
        except Exception:
            body = {}

        return resp.status_code, body, api_ms, request_id

    def get_session(self, session_id: str) -> dict[str, Any]:
        """GET /sessions/{session_id} to query current session status and state."""
        target_url = f"{self.base_url}/sessions/{session_id}"
        resp = self.session.get(target_url, timeout=self.timeout)

        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)

        return resp.json()

    def get_leaderboard(self, game: str = "snake") -> dict[str, Any]:
        """GET /leaderboard/{game} to inspect the leaderboard."""
        target_url = f"{self.base_url}/leaderboard/{game}"
        resp = self.session.get(target_url, timeout=self.timeout)

        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)

        return resp.json()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> BaseGameClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# SnakeClient
# ─────────────────────────────────────────────────────────────────────────────

class SnakeClient(BaseGameClient):
    """Client for TopGameAI Arena Snake."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        agent_name: str = DEFAULT_AGENT_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        allow_provisional: bool = False,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, agent_name=agent_name, timeout=timeout)
        self.allow_provisional: bool = allow_provisional
        self._last_state: dict[str, dict[str, Any]] = {}
        self._last_steps: dict[str, int] = {}
        self._last_response: dict[str, dict[str, Any]] = {}

    def start_game(self, agent_name: str = "") -> SessionStart:
        """Start a Snake arena session.

        POST /sessions with {"game": "snake", "agent_name": agent_name}
        Returns (session_id, initial_state) tuple / SessionStart.
        """
        name = agent_name or self.agent_name
        data = self.create_session(game="snake", agent_name=name)

        session_id = str(data.get("session_id") or data.get("id") or "")
        raw_state = data.get("state", {})
        steps = int(data.get("steps", 0))

        normalized_state = _normalize_state("snake", raw_state, data, session_id=session_id)

        self._last_state[session_id] = normalized_state
        self._last_steps[session_id] = steps
        self._last_response[session_id] = data

        return SessionStart(session_id=session_id, raw_state=normalized_state, raw_response=data)

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        """Submit an action ("up"|"down"|"left"|"right") to Snake arena.

        POST /sessions/{session_id}/action with {"action": action}
        """
        status_code, body, api_ms, req_id = self.submit_action(session_id, action)

        prev_steps = self._last_steps.get(session_id, 0)
        prev_state = state_before or self._last_state.get(session_id, {})

        if status_code == 400:
            # Rejection from HTTP 400 (e.g. 180-degree reversal)
            return MoveResult(
                state_after=prev_state,
                result="rejected",
                game_over=False,
                api_ms=api_ms,
                steps=prev_steps,
                raw_response=body,
                http_status=400,
                won=False,
                score=prev_state.get("score", 0),
                request_id=req_id,
            )

        raw_state = body.get("state", {})
        steps = int(body.get("steps", prev_steps))
        done = bool(body.get("done", False))
        won = bool(body.get("won", False))
        score = int(body.get("score", 0))

        normalized_state = _normalize_state("snake", raw_state, body, session_id=session_id)

        # Rejection detection: if response steps didn't increment, it was rejected
        result: Literal["applied", "rejected"] = "applied"
        if steps <= prev_steps:
            result = "rejected"

        self._last_state[session_id] = normalized_state
        self._last_steps[session_id] = steps
        self._last_response[session_id] = body

        return MoveResult(
            state_after=normalized_state,
            result=result,
            game_over=done,
            api_ms=api_ms,
            steps=steps,
            raw_response=body,
            http_status=200,
            won=won,
            score=score,
            request_id=req_id,
        )

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Fetch current game state via GET /sessions/{session_id}."""
        body = self.get_session(session_id)
        raw_state = body.get("state", body)
        normalized = _normalize_state("snake", raw_state, body, session_id=session_id)
        self._last_state[session_id] = normalized
        self._last_steps[session_id] = int(body.get("steps", self._last_steps.get(session_id, 0)))
        self._last_response[session_id] = body
        return normalized

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        """Retrieve stored final state of the session."""
        if session_id in self._last_state:
            return self._last_state[session_id]
        return self.get_state(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Game2048Client
# ─────────────────────────────────────────────────────────────────────────────

class Game2048Client(BaseGameClient):
    """Client for TopGameAI Arena 2048."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        agent_name: str = DEFAULT_AGENT_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        allow_provisional: bool = False,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, agent_name=agent_name, timeout=timeout)
        self.allow_provisional: bool = allow_provisional
        self._last_state: dict[str, dict[str, Any]] = {}
        self._last_steps: dict[str, int] = {}
        self._last_response: dict[str, dict[str, Any]] = {}

    def start_game(self, agent_name: str = "") -> SessionStart:
        """Start a 2048 arena session.

        POST /sessions with {"game": "2048", "agent_name": agent_name}
        Returns (session_id, initial_state) tuple / SessionStart.
        """
        name = agent_name or self.agent_name
        data = self.create_session(game="2048", agent_name=name)

        session_id = str(data.get("session_id") or data.get("id") or "")
        raw_state = data.get("state", {})
        steps = int(data.get("steps", 0))

        normalized_state = _normalize_state("2048", raw_state, data, session_id=session_id)

        self._last_state[session_id] = normalized_state
        self._last_steps[session_id] = steps
        self._last_response[session_id] = data

        return SessionStart(session_id=session_id, raw_state=normalized_state, raw_response=data)

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        """Submit an action ("up"|"down"|"left"|"right") to 2048 arena.

        POST /sessions/{session_id}/action with {"action": action}
        """
        status_code, body, api_ms, req_id = self.submit_action(session_id, action)

        prev_steps = self._last_steps.get(session_id, 0)
        prev_state = state_before or self._last_state.get(session_id, {})

        if status_code == 400:
            return MoveResult(
                state_after=prev_state,
                result="rejected",
                game_over=False,
                api_ms=api_ms,
                steps=prev_steps,
                raw_response=body,
                http_status=400,
                won=False,
                score=prev_state.get("score", 0),
                request_id=req_id,
            )

        raw_state = body.get("state", {})
        steps = int(body.get("steps", prev_steps))
        done = bool(body.get("done", False))
        won = bool(body.get("won", False))
        score = int(body.get("score", 0))

        normalized_state = _normalize_state("2048", raw_state, body, session_id=session_id)

        # Rejection detection for 2048:
        # 1. Server step count didn't increment
        # 2. Board unchanged (grid before == grid after)
        result: Literal["applied", "rejected"] = "applied"
        if steps <= prev_steps:
            result = "rejected"
        elif prev_state:
            board_before = prev_state.get("grid") or prev_state.get("board")
            board_after = normalized_state.get("grid") or normalized_state.get("board")
            if board_before is not None and board_after is not None and board_before == board_after:
                result = "rejected"

        self._last_state[session_id] = normalized_state
        self._last_steps[session_id] = steps
        self._last_response[session_id] = body

        return MoveResult(
            state_after=normalized_state,
            result=result,
            game_over=done,
            api_ms=api_ms,
            steps=steps,
            raw_response=body,
            http_status=200,
            won=won,
            score=score,
            request_id=req_id,
        )

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Fetch current 2048 state via GET /sessions/{session_id}."""
        body = self.get_session(session_id)
        raw_state = body.get("state", body)
        normalized = _normalize_state("2048", raw_state, body, session_id=session_id)
        self._last_state[session_id] = normalized
        self._last_steps[session_id] = int(body.get("steps", self._last_steps.get(session_id, 0)))
        self._last_response[session_id] = body
        return normalized

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        """Retrieve stored final state of the session."""
        if session_id in self._last_state:
            return self._last_state[session_id]
        return self.get_state(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper Classes (Supporting both Mock Simulation & Live Integration)
# ─────────────────────────────────────────────────────────────────────────────

class SnakeClientWrapper:
    """Wrapper conforming to GameClient protocol, supporting mock simulation."""

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = DEFAULT_AGENT_NAME,
        is_mock: bool = False,
        allow_provisional: bool = False,
    ) -> None:
        self._is_mock = is_mock
        self._agent_name = agent_name
        self._client: SnakeClient | None = (
            None if is_mock else SnakeClient(api_key=api_key, agent_name=agent_name, allow_provisional=allow_provisional)
        )
        self._step_count = 0
        self._snake = [[5, 5], [5, 4], [5, 3]]
        self._food = [8, 5]
        self._direction = "right"
        self._score = 0
        self._game_over = False

    def start_game(self, agent_name: str = "") -> tuple[str, dict[str, Any]]:
        name = agent_name or self._agent_name
        if not self._is_mock and self._client is not None:
            start = self._client.start_game(name)
            return start.session_id, start.raw_state

        self._step_count = 0
        self._snake = [[5, 5], [5, 4], [5, 3]]
        self._food = [8, 5]
        self._direction = "right"
        self._score = 0
        self._game_over = False
        initial_state = {
            "snake": [list(p) for p in self._snake],
            "food": list(self._food),
            "direction": self._direction,
            "score": self._score,
            "game_over": False,
            "step": 0,
        }
        return f"MOCK-SNAKE-{uuid.uuid4().hex[:8]}", initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        if not self._is_mock and self._client is not None:
            return self._client.send_move(session_id, action, state_before)

        self._step_count += 1
        return MoveResult(
            raw_response={"result": "applied"},
            state_after={
                "snake": [list(p) for p in self._snake],
                "food": list(self._food),
                "direction": action,
                "score": self._score,
                "game_over": self._game_over,
                "step": self._step_count,
            },
            result="applied",
            game_over=self._game_over,
            api_ms=1,
            http_status=200,
            steps=self._step_count,
        )

    def get_state(self, session_id: str) -> dict[str, Any]:
        if not self._is_mock and self._client is not None:
            return self._client.get_state(session_id)
        return {
            "snake": [list(p) for p in self._snake],
            "food": list(self._food),
            "direction": self._direction,
            "score": self._score,
            "game_over": self._game_over,
            "step": self._step_count,
        }

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        return self.get_state(session_id)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class Game2048ClientWrapper:
    """Wrapper conforming to GameClient protocol for 2048, supporting mock simulation."""

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = DEFAULT_AGENT_NAME,
        is_mock: bool = False,
        allow_provisional: bool = False,
    ) -> None:
        self._is_mock = is_mock
        self._agent_name = agent_name
        self._client: Game2048Client | None = (
            None if is_mock else Game2048Client(api_key=api_key, agent_name=agent_name, allow_provisional=allow_provisional)
        )
        self._step_count = 0
        self._board = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 2],
        ]
        self._score = 16
        self._game_over = False

    def start_game(self, agent_name: str = "") -> tuple[str, dict[str, Any]]:
        name = agent_name or self._agent_name
        if not self._is_mock and self._client is not None:
            start = self._client.start_game(name)
            return start.session_id, start.raw_state

        self._step_count = 0
        self._board = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 2],
        ]
        self._score = 16
        self._game_over = False
        initial_state = {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": 8,
            "game_over": False,
            "step": 0,
        }
        return f"MOCK-2048-{uuid.uuid4().hex[:8]}", initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        if not self._is_mock and self._client is not None:
            return self._client.send_move(session_id, action, state_before)

        self._step_count += 1
        return MoveResult(
            raw_response={"result": "applied"},
            state_after={
                "board": [row[:] for row in self._board],
                "score": self._score + 4,
                "max_tile": 8,
                "game_over": self._game_over,
                "step": self._step_count,
            },
            result="applied",
            game_over=self._game_over,
            api_ms=1,
            http_status=200,
            steps=self._step_count,
        )

    def get_state(self, session_id: str) -> dict[str, Any]:
        if not self._is_mock and self._client is not None:
            return self._client.get_state(session_id)
        return {
            "board": [row[:] for row in self._board],
            "score": self._score,
            "max_tile": 8,
            "game_over": self._game_over,
            "step": self._step_count,
        }

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        return self.get_state(session_id)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


__all__ = [
    "BaseGameClient",
    "SnakeClient",
    "Game2048Client",
    "MoveResult",
    "SessionStart",
    "GameAPIError",
    "ContractNotDiscoveredError",
    "GameStateView",
    "SnakeClientWrapper",
    "Game2048ClientWrapper",
]
