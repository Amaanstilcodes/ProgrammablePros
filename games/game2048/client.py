"""2048 Client Module — Genesis 2.0 Track EVA 02.

Transport layer for TopGameAI Arena 2048 endpoints.
Raises ContractNotDiscoveredError if endpoints are unverified.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal
from urllib.parse import urljoin

import requests

from core.config import Settings
from core.protocols import (
    ContractNotDiscoveredError,
    GameStateView,
    MoveResult,
    SessionStart,
)
from games.game2048.parser import Game2048StateParser
from games.snake.client import BaseGameClient, GameAPIError

# Discovered path constants (populated ONLY after running tools/probe_api.py)
DISCOVERED_2048_START_PATH: str | None = None
DISCOVERED_2048_MOVE_PATH: str | None = None
DISCOVERED_2048_STATE_PATH: str | None = None


class Game2048Client(BaseGameClient):
    """Contract-driven HTTP client for 2048 arena."""

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
        self._start_path = DISCOVERED_2048_START_PATH or ("/2048/start" if allow_provisional else None)
        self._move_path = DISCOVERED_2048_MOVE_PATH or ("/2048/move" if allow_provisional else None)
        self._state_path = DISCOVERED_2048_STATE_PATH or ("/2048/state" if allow_provisional else None)
        self._parser = Game2048StateParser()

    def start_game(self, agent_name: str = "") -> SessionStart:
        """Start a 2048 arena session."""
        if not self._start_path:
            raise ContractNotDiscoveredError(
                "2048 endpoint paths are not yet verified. "
                "Run 'python tools/probe_api.py --game 2048 --action discover' to capture verified contract fixtures."
            )

        target_url = urljoin(self.base_url + "/", self._start_path.lstrip("/"))
        payload = {"agent": agent_name or self.agent_name, "game": "2048"}

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
        """Send a move to the 2048 arena."""
        if not self._move_path:
            raise ContractNotDiscoveredError(
                "2048 move endpoint path is not yet verified. Run tools/probe_api.py."
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

        # Rejection detection: compare board before vs board after or explicit flag
        result: Literal["applied", "rejected"] = "applied"
        if body.get("result") == "rejected" or body.get("rejected") is True or body.get("no_change") is True:
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
        """Fetch current 2048 game state."""
        if not self._state_path:
            raise ContractNotDiscoveredError(
                "2048 state endpoint path is not yet verified. Run tools/probe_api.py."
            )
        target_url = urljoin(self.base_url + "/", self._state_path.lstrip("/"))
        resp = self.session.get(target_url, params={"session_id": session_id}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise GameAPIError(resp.status_code, target_url, resp.text)
        return resp.json()

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        return self.get_state(session_id)


class Game2048ClientWrapper:
    """Wrapper conforming to GameClient protocol for 2048, supporting mock simulation."""

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
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
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
