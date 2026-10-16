"""Configuration Module — Genesis 2.0 Track EVA 02.

Loads environment variables, .env file, and defines typed configuration dataclasses.
No hardcoded secrets or environment assumptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RejectedStepPolicy = Literal["repeat_last_applied", "pending_ordinal"]


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
class AppConfig:
    """Master application configuration."""

    # Agent identification (Strict rule: MUST be ProgrammablePros)
    agent_name: str = "ProgrammablePros"

    # API endpoints
    topgame_api_url: str = os.getenv("TOPGAME_API_URL", "https://topgameai.com/api")
    topgame_api_key: str = os.getenv("TOPGAME_API_KEY", "")

    # Laya settings
    laya_mode: str = os.getenv("LAYA_MODE", "mock").lower()  # mock | http | local
    laya_api_url: str = os.getenv("LAYA_API_URL", "")
    laya_api_key: str = os.getenv("LAYA_API_KEY", "")

    # Logging & Paths
    log_root: Path = field(default_factory=lambda: Path(os.getenv("LOG_ROOT", "logs")))
    quarantine_mock_dir: Path = field(
        default_factory=lambda: Path(os.getenv("QUARANTINE_DIR", "logs/_mock"))
    )

    # Step ledger policy: "repeat_last_applied" (Default, Policy A) or "pending_ordinal" (Policy B)
    rejected_step_policy: RejectedStepPolicy = os.getenv(
        "REJECTED_STEP_POLICY", "repeat_last_applied"  # type: ignore[assignment]
    )

    # Game step limits per official PDF rules
    snake_max_accepted_steps: int = 300
    game2048_max_accepted_steps: int = 2000

    # Fallback budget limit (Max 10% of total session moves per PDF §4)
    max_fallback_ratio: float = 0.10

    # Verification threshold for Stage 2 unlock (Snake score >= 600)
    snake_unlock_score: int = 600

    @classmethod
    def load(cls, **overrides: Any) -> AppConfig:
        """Create config merging environment variables with optional explicit overrides."""
        return cls(**overrides)
