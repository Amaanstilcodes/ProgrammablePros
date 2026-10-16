"""Game Clients Module — Genesis 2.0 EVA 02 Compliance.

HTTP REST clients for the TopGameAI Arena endpoints:
- Stage 1: Snake  (https://topgameai.com/arena/snake)
- Stage 2: 2048   (https://topgameai.com/arena/2048)

All requests are made via ``requests.Session`` for connection reuse and
automatic cookie persistence (session tokens etc.).

NOTE: Endpoint URLs are structured placeholders matching the site's path
convention. Replace the constants at the top of each client when the
official API documentation is published.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT: float = 15.0  # seconds per request

def _build_session(
    total_retries: int = 3,
    backoff_factor: float = 0.3,
) -> requests.Session:
    """Build a ``requests.Session`` with retry + backoff on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET", "POST"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass
class MoveResult:
    """Structured outcome of a single move sent to a game server.

    Attributes:
        state_after:  Raw JSON state dict returned by the game API after the move.
        result:       "applied" or "rejected" per PDF §2.
        game_over:    True when the game has ended (win or loss).
        api_ms:       Round-trip time (ms) for the POST request itself.
    """
    state_after: dict[str, Any]
    result: str
    game_over: bool
    api_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# BaseGameClient
# ─────────────────────────────────────────────────────────────────────────────

class BaseGameClient:
    """Shared HTTP plumbing for all TopGameAI Arena game clients.

    Subclasses set ``BASE_URL`` and override game-specific methods.
    """

    BASE_URL: str = "https://topgameai.com/api"

    def __init__(
        self,
        api_key: str = "",
        agent_name: str = "ProgrammablePros",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialise the base client.

        Args:
            api_key:    Bearer token or API key for authentication.
            agent_name: Display name submitted with each game registration.
            timeout:    Per-request timeout in seconds.
        """
        self._api_key: str = api_key
        self._agent_name: str = agent_name
        self._timeout: float = timeout
        self._session: requests.Session = _build_session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"ProgrammablePros/1.0 Genesis2-EVA02",
        })

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generic authenticated GET request.

        Args:
            path:   URL path relative to ``BASE_URL``.
            params: Optional query parameters.

        Returns:
            Parsed JSON response dict.

        Raises:
            requests.HTTPError: On 4xx / 5xx responses.
            requests.ConnectionError: On network failure.
        """
        url = f"{self.BASE_URL}{path}"
        resp: Response = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Generic authenticated POST request.

        Args:
            path:    URL path relative to ``BASE_URL``.
            payload: JSON-serialisable request body.

        Returns:
            Parsed JSON response dict.

        Raises:
            requests.HTTPError: On 4xx / 5xx responses.
        """
        url = f"{self.BASE_URL}{path}"
        resp: Response = self._session.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying requests session."""
        self._session.close()

    def __enter__(self) -> "BaseGameClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# SnakeClient
# ─────────────────────────────────────────────────────────────────────────────

class SnakeClient(BaseGameClient):
    """HTTP client for the TopGameAI Snake arena (Stage 1).

    Endpoints (update paths when official docs are released):
        POST /snake/start  — Register agent, receive session_id + initial state
        POST /snake/move   — Send direction, receive new state
        GET  /snake/state  — Poll current state (e.g., final state on game-over)
    """

    # ── Endpoint paths (trivially updated when docs arrive) ──────────────────
    _PATH_START: str = "/snake/start"
    _PATH_MOVE: str = "/snake/move"
    _PATH_STATE: str = "/snake/state"

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        """Register a new Snake game session.

        Args:
            agent_name: Override the default agent display name for this session.

        Returns:
            Tuple of (session_id: str, initial_state: dict).

        Raises:
            KeyError: If the API response is missing ``session_id`` or ``state``.
        """
        payload: dict[str, Any] = {
            "agent": agent_name or self._agent_name,
            "game": "snake",
        }
        resp = self._post(self._PATH_START, payload)

        session_id: str = str(resp["session_id"])
        # Normalise: the initial game state may be nested under "state" or "initial_state"
        initial_state: dict[str, Any] = resp.get("state") or resp.get("initial_state") or resp
        return session_id, initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
    ) -> MoveResult:
        """Send a directional move and return the structured outcome.

        Detects ``"rejected"`` moves by checking whether the API explicitly
        flags rejection OR whether the game state is unchanged (snake did not
        grow/move).

        Args:
            session_id: Opaque session token from ``start_game()``.
            action:     Direction string: "up" | "down" | "left" | "right".

        Returns:
            ``MoveResult`` with state_after, result, game_over, api_ms.
        """
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action": action.strip().lower(),
        }

        t0 = time.perf_counter()
        resp = self._post(self._PATH_MOVE, payload)
        api_ms = (time.perf_counter() - t0) * 1000.0

        # Extract state_after (API may wrap it or return flat)
        state_after: dict[str, Any] = resp.get("state") or resp.get("state_after") or resp

        # Determine result
        raw_result: str = str(
            resp.get("result") or resp.get("status") or "applied"
        ).lower()
        result: str = "rejected" if raw_result in ("rejected", "invalid", "illegal") else "applied"

        # Detect game over from multiple possible API conventions
        game_over: bool = bool(
            resp.get("game_over")
            or resp.get("done")
            or resp.get("finished")
            or state_after.get("game_over")
            or state_after.get("done")
        )

        return MoveResult(
            state_after=state_after,
            result=result,
            game_over=game_over,
            api_ms=round(api_ms, 3),
        )

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        """Fetch the terminal game state after the session has ended.

        Args:
            session_id: Opaque session token.

        Returns:
            Raw final state dict from the API.
        """
        resp = self._get(self._PATH_STATE, params={"session_id": session_id})
        return resp.get("state") or resp.get("final_state") or resp


# ─────────────────────────────────────────────────────────────────────────────
# Game2048Client
# ─────────────────────────────────────────────────────────────────────────────

class Game2048Client(BaseGameClient):
    """HTTP client for the TopGameAI 2048 arena (Stage 2).

    A move in 2048 is "rejected" (no board change) when all tiles in the
    requested direction are already fully merged or blocked.  We detect this
    by comparing board state before and after the move.

    Endpoints (update paths when official docs are released):
        POST /2048/start  — Register agent, receive session_id + initial board
        POST /2048/move   — Send direction, receive new board state
        GET  /2048/state  — Poll current state
    """

    _PATH_START: str = "/2048/start"
    _PATH_MOVE: str = "/2048/move"
    _PATH_STATE: str = "/2048/state"

    def start_game(self, agent_name: str | None = None) -> tuple[str, dict[str, Any]]:
        """Register a new 2048 game session.

        Args:
            agent_name: Override agent display name.

        Returns:
            Tuple of (session_id: str, initial_state: dict).
        """
        payload: dict[str, Any] = {
            "agent": agent_name or self._agent_name,
            "game": "2048",
        }
        resp = self._post(self._PATH_START, payload)
        session_id: str = str(resp["session_id"])
        initial_state: dict[str, Any] = resp.get("state") or resp.get("initial_state") or resp
        return session_id, initial_state

    def send_move(
        self,
        session_id: str,
        action: str,
        state_before: dict[str, Any] | None = None,
    ) -> MoveResult:
        """Send a directional swipe and return the structured outcome.

        In 2048, a move that does not change any tile is officially "rejected."
        We detect this either from the API's explicit rejection flag, or by
        comparing board matrices when ``state_before`` is supplied.

        Args:
            session_id:    Opaque session token.
            action:        Direction: "up" | "down" | "left" | "right".
            state_before:  Optional state dict from before the move, used to
                           detect board-unchanged rejection.

        Returns:
            ``MoveResult`` with state_after, result, game_over, api_ms.
        """
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action": action.strip().lower(),
        }

        t0 = time.perf_counter()
        resp = self._post(self._PATH_MOVE, payload)
        api_ms = (time.perf_counter() - t0) * 1000.0

        state_after: dict[str, Any] = resp.get("state") or resp.get("state_after") or resp

        # Check explicit rejection signal from server
        raw_result: str = str(
            resp.get("result") or resp.get("status") or "applied"
        ).lower()
        if raw_result in ("rejected", "invalid", "no_change", "unchanged"):
            result = "rejected"
        elif state_before is not None:
            # Detect silent rejection: board unchanged after move
            board_before = state_before.get("board") or state_before.get("grid")
            board_after = state_after.get("board") or state_after.get("grid")
            if board_before is not None and board_after is not None:
                result = "rejected" if board_before == board_after else "applied"
            else:
                result = "applied"
        else:
            result = "applied"

        game_over: bool = bool(
            resp.get("game_over")
            or resp.get("done")
            or resp.get("finished")
            or state_after.get("game_over")
            or state_after.get("done")
        )

        return MoveResult(
            state_after=state_after,
            result=result,
            game_over=game_over,
            api_ms=round(api_ms, 3),
        )

    def get_final_state(self, session_id: str) -> dict[str, Any]:
        """Fetch terminal state after the game has ended.

        Args:
            session_id: Opaque session token.

        Returns:
            Raw final state dict.
        """
        resp = self._get(self._PATH_STATE, params={"session_id": session_id})
        return resp.get("state") or resp.get("final_state") or resp


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (mock only — no live network calls)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from unittest.mock import MagicMock, patch

    # ── Patch requests.Session.post to avoid real HTTP calls ─────────────────
    def _mock_snake_start(*args: Any, **kwargs: Any) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "session_id": "SNAKE-MOCK-001",
            "state": {
                "board": [[0] * 10 for _ in range(10)],
                "direction": "right",
                "score": 0,
                "game_over": False,
            },
        }
        return mock_resp

    def _mock_snake_move(*args: Any, **kwargs: Any) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "result": "applied",
            "state": {
                "board": [[0] * 10 for _ in range(10)],
                "direction": "up",
                "score": 10,
                "game_over": False,
            },
        }
        return mock_resp

    with patch("requests.Session.post", side_effect=[_mock_snake_start(), _mock_snake_move()]):
        client = SnakeClient(api_key="test-key")
        sid, init_state = client.start_game("TestAgent")
        assert sid == "SNAKE-MOCK-001"
        assert "board" in init_state
        print(f"[SnakeClient] start_game PASSED — session_id={sid}")

        move_result = client.send_move(sid, "up")
        assert move_result.result == "applied"
        assert move_result.game_over is False
        assert move_result.api_ms >= 0.0
        print(f"[SnakeClient] send_move PASSED — result={move_result.result}, api_ms={move_result.api_ms:.2f}ms")
        client.close()

    print("\n[GameClients] ALL SELF-TESTS PASSED")
