"""Configuration & Runtime Settings — Genesis 2.0 Track EVA 02.

Provides frozen, validated Settings loaded from environment variables and .env file.
Enforces mock quarantine and live safety rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RejectedStepPolicy = Literal["repeat_last_applied", "pending_ordinal"]
LayaMode = Literal["mock", "local", "http"]
HypothesisMode = Literal["full", "lean"]


def _load_env_file(dotenv_path: Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    path = dotenv_path or Path(".env")
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_env_file()


@dataclass(frozen=True)
class Settings:
    """Master application configuration and runtime safety constraints."""

    agent_name: str = "ProgrammablePros"
    topgameai_base_url: str = os.getenv("TOPGAMEAI_BASE_URL", os.getenv("TOPGAME_API_URL", "https://topgameai.com/api"))
    topgameai_api_key: str = os.getenv("TOPGAMEAI_API_KEY", os.getenv("TOPGAME_API_KEY", ""))

    laya_mode: LayaMode = os.getenv("LAYA_MODE", "mock").lower()  # type: ignore[assignment]
    laya_model_path: str = os.getenv("LAYA_MODEL_PATH", "")
    laya_endpoint: str = os.getenv("LAYA_ENDPOINT", os.getenv("LAYA_API_URL", ""))
    laya_api_key: str = os.getenv("LAYA_API_KEY", "")

    log_root: str = os.getenv("LOG_ROOT", "logs")
    rejected_step_policy: RejectedStepPolicy = os.getenv(  # type: ignore[assignment]
        "REJECTED_STEP_POLICY", "repeat_last_applied"
    )
    max_fallback_ratio: float = 0.10
    probability_decimals: int = 6
    hypothesis_mode: HypothesisMode = os.getenv("HYPOTHESIS_MODE", "full").lower()  # type: ignore[assignment]

    snake_max_steps: int = 300
    game2048_max_steps: int = 2000

    http_timeout_s: float = 10.0
    http_retries: int = 3
    http_backoff_s: float = 0.3

    mock_quarantine: bool = True  # Hard enforced
    unlock_confirmed: bool = False
    snake_proof_path: str | None = None

    # Backward compatibility properties
    @property
    def topgame_api_url(self) -> str:
        return self.topgameai_base_url

    @property
    def topgame_api_key(self) -> str:
        return self.topgameai_api_key

    @property
    def snake_max_accepted_steps(self) -> int:
        return self.snake_max_steps

    @property
    def game2048_max_accepted_steps(self) -> int:
        return self.game2048_max_steps

    @property
    def quarantine_mock_dir(self) -> Path:
        return Path(self.log_root) / "_mock"

    @property
    def snake_unlock_score(self) -> int:
        return 600

    def require_live_safety(
        self,
        game: str = "snake",
        is_self_test: bool = False,
        target_log_path: Path | None = None,
    ) -> None:
        """Enforce strict isolation between mock testing and live competition.

        Raises:
            RuntimeError: if safety constraints are violated.
        """
        # Rule 1: mock mode must be self-test
        if self.laya_mode == "mock" and not is_self_test:
            raise RuntimeError(
                "SAFETY VIOLATION: LAYA_MODE='mock' is only permitted during explicit self-test runs."
            )

        # Rule 2: log path outside logs/_mock while laya_mode == 'mock'
        if self.laya_mode == "mock" and target_log_path is not None:
            resolved_target = str(target_log_path.resolve())
            resolved_mock = str(self.quarantine_mock_dir.resolve())
            if not resolved_target.startswith(resolved_mock):
                raise RuntimeError(
                    f"SAFETY VIOLATION: Mock logs must be quarantined in {self.quarantine_mock_dir}, got {target_log_path}"
                )

        # Rule 3: 2048 requires unlock confirmation
        if game == "2048" and not self.unlock_confirmed:
            raise RuntimeError(
                "SAFETY VIOLATION: Cannot execute Stage 2 (2048) without explicit unlock_confirmed=True."
            )

    def redacted(self) -> dict[str, Any]:
        """Return dict representation with all credentials redacted."""
        return {
            "agent_name": self.agent_name,
            "topgameai_base_url": self.topgameai_base_url,
            "topgameai_api_key": "[REDACTED]" if self.topgameai_api_key else "",
            "laya_mode": self.laya_mode,
            "laya_model_path": self.laya_model_path,
            "laya_endpoint": self.laya_endpoint,
            "laya_api_key": "[REDACTED]" if self.laya_api_key else "",
            "log_root": self.log_root,
            "rejected_step_policy": self.rejected_step_policy,
            "max_fallback_ratio": self.max_fallback_ratio,
            "probability_decimals": self.probability_decimals,
            "hypothesis_mode": self.hypothesis_mode,
            "snake_max_steps": self.snake_max_steps,
            "game2048_max_steps": self.game2048_max_steps,
            "http_timeout_s": self.http_timeout_s,
            "mock_quarantine": self.mock_quarantine,
            "unlock_confirmed": self.unlock_confirmed,
        }

    @classmethod
    def load(cls, **overrides: Any) -> Settings:
        """Load settings with optional runtime overrides."""
        return cls(**overrides)


# Alias AppConfig to Settings for full backwards compatibility
AppConfig = Settings
