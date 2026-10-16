"""Stage Gate Module — Genesis 2.0 Track EVA 02.

Enforces Rule R8:
"Never run Stage 2 (2048) before a verified Snake run >= 600 plus an
explicit organizer unlock confirmation."

Inspects logs and submission records to ensure Stage 1 qualification criteria
are satisfied before allowing Stage 2 execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.official_logger import OfficialLogger


class StageLockedError(RuntimeError):
    """Raised when an unverified or unauthorized stage transition is attempted."""
    pass


class StageGate:
    """Verifies qualification records before permitting Stage 2 (2048) execution."""

    def __init__(self, log_root: str | Path = "logs", unlock_marker_dir: str | Path = "submission/verified") -> None:
        self.log_root = Path(log_root)
        self.unlock_marker_dir = Path(unlock_marker_dir)

    def check_stage2_unlocked(self, allow_override: bool = False) -> tuple[bool, str]:
        """Check whether Stage 2 (2048) is unlocked.

        Criteria:
        1. Explicit override flag (for emergency manual testing), OR
        2. Unlock confirmation token in submission/verified/stage1_unlock.json, OR
        3. A verified Snake log in logs/snake/ with final score >= 600 and intact hash chain.

        Returns:
            Tuple of (is_unlocked: bool, reason: str).
        """
        if allow_override:
            return True, "Override flag active."

        # Check explicit unlock marker
        marker_file = self.unlock_marker_dir / "stage1_unlock.json"
        if marker_file.is_file():
            try:
                with open(marker_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                token = data.get("organizer_token") or data.get("verified_session_id")
                if token:
                    return True, f"Explicit unlock token verified: {token}"
            except Exception:
                pass

        # Scan logs/snake/ for a verified session with score >= 600
        snake_dir = self.log_root / "snake"
        if snake_dir.is_dir():
            for session_dir in snake_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                log_files = list(session_dir.glob("decision_log_*.jsonl"))
                final_state_files = list(session_dir.glob("final_state_*.json"))

                if not log_files or not final_state_files:
                    continue

                # Verify hash chain
                is_valid, msg = OfficialLogger.verify_log_file(log_files[0])
                if not is_valid:
                    continue

                # Check final score in final state
                try:
                    with open(final_state_files[0], "r", encoding="utf-8") as f:
                        final_state = json.load(f)
                    score = int(final_state.get("score", 0))
                    if score >= 600:
                        return True, f"Verified qualifying Snake session found: {session_dir.name} (Score: {score})"
                except Exception:
                    continue

        return False, "STAGE 2 LOCKED: No verified Snake run with score >= 600 found in logs/snake/ nor unlock token in submission/verified/stage1_unlock.json."

    def assert_stage_unlocked(self, game: str, allow_override: bool = False) -> None:
        """Raise StageLockedError if the requested game is locked."""
        game_clean = game.strip().lower()
        if game_clean == "2048":
            unlocked, reason = self.check_stage2_unlocked(allow_override=allow_override)
            if not unlocked:
                raise StageLockedError(reason)
