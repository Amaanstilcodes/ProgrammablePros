"""Official Logger Module — Genesis 2.0 EVA 02 Compliance.

Implements the exact JSONL logging schema mandated by the Genesis 2.0 Track PDF:
- One line per APPLIED move, one line per REJECTED move
- Cryptographic SHA-256 hash chain: hash = SHA256(prev_hash + "|" + canonical_json)
- Canonical JSON: sorted keys, no spaces, UTF-8, no NaN / Infinity
- Final game state persisted to final_state_{session_id}.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_json(data: Any) -> str:
    """Produce a deterministic, UTF-8 safe JSON string.

    Rules (from PDF §Hash Chain):
    - Keys sorted recursively.
    - No spaces between tokens (``separators=(',', ':')``).
    - No NaN or Infinity — raises ValueError if found.
    """
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                raise ValueError(
                    f"Canonical JSON forbids NaN/Infinity. Got: {obj!r}"
                )
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    sanitized = _sanitize(data)
    return json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_hash(line_data: dict[str, Any], prev_hash: str) -> str:
    """Compute SHA-256 for a log line.

    Hash input: ``prev_hash + "|" + canonical_json(line_without_hash_field)``

    Args:
        line_data: Complete log line dict WITHOUT the ``hash`` key.
        prev_hash: SHA-256 hex string of the previous entry (or seed string).

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    payload = prev_hash + "|" + _canonical_json(line_data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string with millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ─────────────────────────────────────────────────────────────────────────────
# OfficialLogger
# ─────────────────────────────────────────────────────────────────────────────

class OfficialLogger:
    """Append-only JSONL logger that enforces the Genesis 2.0 hash chain schema.

    Usage:
        logger = OfficialLogger(log_dir="logs")
        logger.start_session(session_id="abc123", game_name="snake")
        logger.log_move(
            session_id="abc123",
            game="snake",
            step=1,
            state_before={...},
            action="up",
            state_after={...},
            result="applied",
            decision={...},
            timing={...},
        )
        logger.save_final_state("abc123", final_state_dict={...})
    """

    def __init__(self, log_dir: str = ".") -> None:
        self._log_dir: Path = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Keyed by session_id
        self._prev_hashes: dict[str, str] = {}
        self._step_counters: dict[str, int] = {}
        self._file_handles: dict[str, Any] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(self, session_id: str, game_name: str) -> Path:
        """Initialise a new logging session.

        Sets the genesis ``prev_hash`` to ``"GENESIS-{session_id}"`` as per PDF §5.

        Args:
            session_id: Unique opaque session identifier returned by game API.
            game_name: "snake" or "2048".

        Returns:
            Absolute path to the opened ``.jsonl`` log file.
        """
        log_path = self._log_dir / f"decision_log_{session_id}.jsonl"
        # Close any existing handle for this session before reopening
        if session_id in self._file_handles:
            fh = self._file_handles.pop(session_id)
            if not fh.closed:
                fh.close()

        self._prev_hashes[session_id] = f"GENESIS-{session_id}"
        self._step_counters[session_id] = 0
        self._file_handles[session_id] = open(log_path, "a", encoding="utf-8")
        return log_path.resolve()

    def close_session(self, session_id: str) -> None:
        """Flush and close the file handle for a session."""
        fh = self._file_handles.pop(session_id, None)
        if fh and not fh.closed:
            fh.flush()
            fh.close()

    # ── Core logging method ───────────────────────────────────────────────────

    def log_move(
        self,
        *,
        session_id: str,
        game: str,
        step: int,
        state_before: dict[str, Any],
        action: str,
        state_after: dict[str, Any],
        result: str,
        decision: dict[str, Any],
        timing: dict[str, float],
        ts: str | None = None,
    ) -> dict[str, Any]:
        """Append one move line to the JSONL log with a valid SHA-256 hash.

        Args:
            session_id:   Opaque session identifier.
            game:         "snake" or "2048".
            step:         Integer step index (no gaps allowed).
            state_before: Exact game state dict received BEFORE this move.
            action:       Direction string sent to the game API.
            state_after:  Exact game state dict received AFTER this move.
            result:       "applied" or "rejected".
            decision:     Exact decision block dict (built by ComplianceLayer).
            timing:       Dict with keys decide_ms, api_ms, turn_ms (floats).
            ts:           ISO 8601 UTC timestamp. Auto-generated if None.

        Returns:
            The fully constructed log line dict (including hash).

        Raises:
            RuntimeError: If session has not been started.
            ValueError: If any value would violate canonical JSON rules.
        """
        if session_id not in self._prev_hashes:
            raise RuntimeError(
                f"Session '{session_id}' not initialised. Call start_session() first."
            )

        prev_hash = self._prev_hashes[session_id]

        line: dict[str, Any] = {
            "session_id": session_id,
            "game": game,
            "step": step,
            "state_before": state_before,
            "action": action,
            "state_after": state_after,
            "result": result,
            "decision": decision,
            "timing": timing,
            "ts": ts or _utc_now_iso(),
            "prev_hash": prev_hash,
        }

        # Compute hash over the line WITHOUT the "hash" field (PDF §5)
        new_hash = _compute_hash(line, prev_hash)
        line["hash"] = new_hash

        # Persist
        fh = self._file_handles.get(session_id)
        if fh is None or fh.closed:
            raise RuntimeError(
                f"File handle for session '{session_id}' is closed. "
                "Call start_session() before logging."
            )
        fh.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()

        # Advance chain
        self._prev_hashes[session_id] = new_hash
        self._step_counters[session_id] = step

        return line

    # ── Final state persistence ───────────────────────────────────────────────

    def save_final_state(
        self,
        session_id: str,
        final_state_dict: dict[str, Any],
    ) -> Path:
        """Write final game state to ``final_state_{session_id}.json`` (PDF §7).

        Args:
            session_id:       Opaque session identifier.
            final_state_dict: Raw final-state JSON returned by the game API.

        Returns:
            Absolute path to the written file.
        """
        out_path = self._log_dir / f"final_state_{session_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_state_dict, f, ensure_ascii=False, indent=2)
        return out_path.resolve()

    # ── Verification utility (for __main__ / CI) ─────────────────────────────

    @staticmethod
    def verify_log_file(log_path: str | Path) -> tuple[bool, str]:
        """Re-compute and verify every hash in a completed JSONL log file.

        Args:
            log_path: Path to the ``.jsonl`` file.

        Returns:
            Tuple of (is_valid: bool, message: str).
        """
        path = Path(log_path)
        if not path.exists():
            return False, f"Log file not found: {path}"

        lines: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    return False, f"Line {lineno}: JSON parse error — {exc}"

        if not lines:
            return False, "Log file is empty."

        for idx, line in enumerate(lines):
            stored_hash = line.get("hash", "")
            stored_prev = line.get("prev_hash", "")

            # Reconstruct line without hash to recompute
            line_without_hash = {k: v for k, v in line.items() if k != "hash"}
            expected_hash = _compute_hash(line_without_hash, stored_prev)

            if stored_hash != expected_hash:
                return (
                    False,
                    f"Hash mismatch at step {line.get('step', idx)}: "
                    f"stored={stored_hash[:16]}… expected={expected_hash[:16]}…",
                )

            # Verify chain continuity: prev_hash[n] == hash[n-1]
            if idx > 0:
                expected_prev = lines[idx - 1]["hash"]
                if stored_prev != expected_prev:
                    return (
                        False,
                        f"Chain break at step {line.get('step', idx)}: "
                        f"prev_hash mismatch.",
                    )

        return True, f"Hash chain VALID — {len(lines)} entries verified."


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = OfficialLogger(log_dir=tmpdir)
        sid = "TEST-SESSION-001"
        logger.start_session(sid, "snake")

        for step_i in range(1, 4):
            state_a = {"board": [[0] * 4 for _ in range(4)], "step": step_i - 1}
            state_b = {"board": [[0] * 4 for _ in range(4)], "step": step_i}
            decision = {
                "questions": [f"hypothesis_q{step_i}"],
                "probabilities": {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25},
                "chosen": "up",
                "confidence": 0.25,
                "source": "laya",
            }
            timing = {"decide_ms": 12.3, "api_ms": 8.7, "turn_ms": 21.0}
            logger.log_move(
                session_id=sid,
                game="snake",
                step=step_i,
                state_before=state_a,
                action="up",
                state_after=state_b,
                result="applied",
                decision=decision,
                timing=timing,
            )

        final = {"final_score": 42, "game_over": True}
        logger.save_final_state(sid, final)
        logger.close_session(sid)

        from pathlib import Path as _P
        log_file = _P(tmpdir) / f"decision_log_{sid}.jsonl"
        valid, msg = OfficialLogger.verify_log_file(log_file)
        assert valid, f"Self-test FAILED: {msg}"
        print("[OfficialLogger self-test PASSED]")
        print(f"  {msg}")
