"""Snake Client Module — Genesis 2.0 Track EVA 02.

Transport layer for TopGameAI Arena Snake endpoints.
Strictly contract-driven: raises ContractNotDiscoveredError if endpoints are unverified.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config import Settings
from core.protocols import (
    ContractNotDiscoveredError,
    GameStateView,
    MoveResult,
    SessionStart,
)
from games.snake.parser import SnakeStateParser

# Discovered path constants (populated ONLY after running tools/probe_api.py)
DISCOVERED_SNAKE_START_PATH: str | None = None
DISCOVERED_SNAKE_MOVE_PATH: str | None = None
DISCOVERED_SNAKE_STATE_PATH: str | None = None


class GameAPIError(RuntimeError):
    """Raised on unexpected HTTP status or transport error."""

    def __init__(self, status_code: int, endpoint: str, body: str) -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        self.body = body
        super().__init__(f"Game API error at {endpoint} [HTTP {status_code}]: {body[:200]}")


def _build_session(retries: int = 3, backoff: float = 0.3) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class BaseGameClient:
    """Shared HTTP transport client for TopGameAI Arena."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
        timeout: float = 10.0,
    ) -> None:
        settings = Settings.load()
        self.base_url = (base_url or settings.topgameai_base_url).rstrip("/")
        self.api_key = api_key or settings.topgameai_api_key
        self.agent_name = agent_name or settings.agent_name
        self.timeout = timeout or settings.http_timeout_s

        self.session = _build_session(retries=settings.http_retries, backoff=settings.http_backoff_s)
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"ProgrammablePros/1.0 Genesis2-EVA02",
        })
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> BaseGameClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class SnakeClient(BaseGameClient):
    """Contract-driven HTTP client for Snake arena."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
        timeout: float = 10.0,
        allow_provisional: bool = False,
    ) -> None:
        super().__init__(base_url, api_key, agent_name, timeout)
        self.allow_provisional = allow_provisional
        self._start_path = DISCOVERED_SNAKE_START_PATH or ("/snake/start" if allow_provisional else None)
        self._move_path = DISCOVERED_SNAKE_MOVE_PATH or ("/snake/move" if allow_provisional else None)
        self._state_path = DISCOVERED_SNAKE_STATE_PATH or ("/snake/state" if allow_provisional else None)
        self._parser = SnakeStateParser()

    def start_game(self, agent_name: str = "") -> SessionStart:
        """Start a Snake arena session."""
        if not self._start_path:
            raise ContractNotDiscoveredError(
                "Snake endpoint paths are not yet verified. "
                "Run 'python tools/probe_api.py --game snake --action discover' to capture verified contract fixtures."
            )

        target_url = urljoin(self.base_url + "/", self._start_path.lstrip("/"))
        payload = {"agent": agent_name or self.agent_name, "game": "snake"}

        t0 = time.perf_counter()
        resp = self.session.post(target_url, json=payload, timeout=self.timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)

        body = resp.json()
        session_id = str(body.get("session_id") or body.get("id") or "")
        raw_state = body.get("state", body)

        return SessionStart(session_id=session_id, raw_state=raw_state, raw_response=body)

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        """Send a move to the Snake arena."""
        if not self._move_path:
            raise ContractNotDiscoveredError(
                "Snake move endpoint path is not yet verified. Run tools/probe_api.py."
            )

        target_url = urljoin(self.base_url + "/", self._move_path.lstrip("/"))
        payload = {"session_id": session_id, "action": action}

        t0 = time.perf_counter()
        resp = self.session.post(target_url, json=payload, timeout=self.timeout)
        api_ms = int((time.perf_counter() - t0) * 1000.0)

        if resp.status_code >= 500:
            raise GameAPIError(resp.status_code, target_url, resp.text)

        body = resp.json() if resp.status_code == 200 else {}
        state_after = body.get("state", body)
        game_over = bool(body.get("game_over") or body.get("done") or False)

        # Rejection detection
        result: Literal["applied", "rejected"] = "applied"
        if body.get("result") == "rejected" or body.get("rejected") is True:
            result = "rejected"
        elif resp.status_code in (400, 422):
            result = "rejected"
        elif state_before is not None:
            if self._parser.is_rejected(state_before, state_after, body):
                result = "rejected"

        return MoveResult(
            raw_response=body,
            state_after=state_after,
            result=result,
            game_over=game_over,
            api_ms=api_ms,
            http_status=resp.status_code,
            request_id=resp.headers.get("x-request-id"),
        )

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Fetch current Snake game state."""
        if not self._state_path:
            raise ContractNotDiscoveredError(
                "Snake state endpoint path is not yet verified. Run tools/probe_api.py."
            )
        target_url = urljoin(self.base_url + "/", self._state_path.lstrip("/"))
        resp = self.session.get(target_url, params={"session_id": session_id}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)
        return resp.json()

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        return self.get_state(session_id)


class SnakeClientWrapper:
    """Wrapper conforming to GameClient protocol, supporting mock simulation."""

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
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

        # Mock transition
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
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
