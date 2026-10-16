"""Official Logger Module — Genesis 2.0 Track EVA 02.

Strict adherence to the official competition JSONL logging specification:
- Dual logging path: logs/<game>/<session_id>/decision_log_<session_id>.jsonl
- Offline mock runs quarantined under logs/_mock/<session_id>/
- SHA-256 cryptographic hash chain starting with GENESIS-<session_id>
- Canonical JSON encoding (sorted keys, no whitespace, no NaN / Infinity, UTF-8)
- Immediate flush per line
- Atomic writes (tempfile + os.replace) for final_state and manifest.json
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    """Produce deterministic canonical JSON string per PDF §Hash Chain.

    Rules:
    - Keys sorted recursively.
    - No spaces between tokens (separators=(',', ':')).
    - No NaN, Infinity, or -Infinity (allow_nan=False).
    - UTF-8 characters preserved (ensure_ascii=False).
    """
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                raise ValueError(f"Canonical JSON forbids NaN/Infinity. Found: {obj!r}")
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    sanitized = _sanitize(data)
    return json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_sha256_hash(line_data: dict[str, Any], prev_hash: str) -> str:
    """Compute SHA-256 hash for a log line.

    Hash input: prev_hash + "|" + canonical_json(line_without_hash)
    """
    payload = prev_hash + "|" + canonical_json(line_data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_git_commit_hash() -> str:
    """Get current HEAD git commit hash or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2.0
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class OfficialLogger:
    """Append-only cryptographic JSONL logger strictly implementing PDF §Hash Chain."""

    def __init__(
        self,
        base_dir: str | Path = "logs",
        is_mock: bool = False,
    ) -> None:
        self._base_dir: Path = Path(base_dir)
        self._is_mock: bool = is_mock
        self._prev_hash: str | None = None
        self._session_id: str | None = None
        self._game: str | None = None
        self._session_dir: Path | None = None
        self._log_file_path: Path | None = None
        self._file_handle: Any = None
        self._start_ts: str | None = None
        self._step_counter: int = 0
        self._accepted_counter: int = 0
        self._fallback_counter: int = 0

    @property
    def log_path(self) -> Path | None:
        return self._log_file_path

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start_session(
        self,
        session_id: str,
        game: str,
        agent_name: str = "ProgrammablePros",
        custom_log_path: Path | None = None,
    ) -> Path:
        """Initialise a session directory and log file.

        Path layout:
          Official:   logs/<game>/<session_id>/decision_log_<session_id>.jsonl
          Quarantine: logs/_mock/<session_id>/decision_log_<session_id>.jsonl
        """
        self._session_id = session_id
        self._game = game.strip().lower()
        self._start_ts = utc_now_iso()
        self._prev_hash = f"GENESIS-{session_id}"
        self._step_counter = 0
        self._accepted_counter = 0
        self._fallback_counter = 0

        if custom_log_path is not None:
            self._log_file_path = Path(custom_log_path)
            self._session_dir = self._log_file_path.parent
            self._session_dir.mkdir(parents=True, exist_ok=True)
        else:
            if self._is_mock:
                self._session_dir = self._base_dir / "_mock" / session_id
            else:
                self._session_dir = self._base_dir / self._game / session_id
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._log_file_path = self._session_dir / f"decision_log_{session_id}.jsonl"

        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()

        self._file_handle = open(self._log_file_path, "a", encoding="utf-8")
        return self._log_file_path.resolve()

    def log_move(
        self,
        *,
        step: int,
        state_before: dict[str, Any],
        action: str,
        state_after: dict[str, Any],
        result: str,
        decision: dict[str, Any],
        timing: dict[str, int],
        ts: str | None = None,
    ) -> dict[str, Any]:
        """Write an official move record to the JSONL log with hash chain computation.

        Enforces deep copying of raw API state objects (Invariant I9).
        """
        if self._file_handle is None or self._file_handle.closed:
            raise RuntimeError("Logger session not active. Call start_session() first.")
        if self._prev_hash is None or self._session_id is None or self._game is None:
            raise RuntimeError("Session state uninitialized.")

        # Invariant I9: Unmodified deep copies of raw API objects
        safe_state_before = copy.deepcopy(state_before)
        safe_state_after = copy.deepcopy(state_after)

        line_without_hash: dict[str, Any] = {
            "session_id": self._session_id,
            "game": self._game,
            "step": int(step),
            "state_before": safe_state_before,
            "action": str(action).strip().lower(),
            "state_after": safe_state_after,
            "result": str(result).strip().lower(),
            "decision": copy.deepcopy(decision),
            "timing": {
                "decide_ms": int(timing.get("decide_ms", 0)),
                "api_ms": int(timing.get("api_ms", 0)),
                "turn_ms": int(timing.get("turn_ms", 0)),
            },
            "ts": ts or utc_now_iso(),
            "prev_hash": self._prev_hash,
        }

        # SHA-256 over line WITHOUT the "hash" field (PDF §5)
        new_hash = compute_sha256_hash(line_without_hash, self._prev_hash)
        full_line = dict(line_without_hash)
        full_line["hash"] = new_hash

        # Write canonical JSON + newline, immediate flush
        serialized = canonical_json(full_line)
        self._file_handle.write(serialized + "\n")
        self._file_handle.flush()

        # Update chain state
        self._prev_hash = new_hash
        self._step_counter = step
        if result == "applied":
            self._accepted_counter += 1
        if decision.get("source") == "fallback":
            self._fallback_counter += 1

        return full_line

    def save_final_state(self, final_state: dict[str, Any]) -> Path:
        """Atomically persist final game state to final_state_<session_id>.json."""
        if self._session_dir is None or self._session_id is None:
            raise RuntimeError("Session directory uninitialized.")

        target_file = self._session_dir / f"final_state_{self._session_id}.json"
        safe_state = copy.deepcopy(final_state)

        # Atomic write: write to temp file then rename
        with tempfile.NamedTemporaryFile("w", dir=self._session_dir, delete=False, encoding="utf-8") as tf:
            json.dump(safe_state, tf, indent=2, ensure_ascii=False)
            temp_name = tf.name

        os.replace(temp_name, target_file)
        return target_file.resolve()

    def write_manifest(
        self,
        agent_name: str = "ProgrammablePros",
        outcome: str = "completed",
        abandonment_reason: str | None = None,
        unlock_ref: str | None = None,
        laya_revision: str = "mock",
    ) -> Path:
        """Atomically write manifest.json summarizing session metadata."""
        if self._session_dir is None or self._session_id is None:
            raise RuntimeError("Session directory uninitialized.")

        target_file = self._session_dir / "manifest.json"
        manifest_data = {
            "session_id": self._session_id,
            "game": self._game,
            "agent_name": agent_name,
            "start_ts": self._start_ts,
            "end_ts": utc_now_iso(),
            "outcome": outcome,
            "abandonment_reason": abandonment_reason,
            "unlock_reference": unlock_ref,
            "git_commit": _get_git_commit_hash(),
            "laya_model_revision": laya_revision,
            "is_mock": self._is_mock,
            "accepted_moves": self._accepted_counter,
            "fallback_moves": self._fallback_counter,
            "fallback_ratio": (
                (self._fallback_counter / self._accepted_counter)
                if self._accepted_counter > 0
                else 0.0
            ),
        }

        with tempfile.NamedTemporaryFile("w", dir=self._session_dir, delete=False, encoding="utf-8") as tf:
            json.dump(manifest_data, tf, indent=2, ensure_ascii=False)
            temp_name = tf.name

        os.replace(temp_name, target_file)
        return target_file.resolve()

    def close(self) -> None:
        """Close log file handle."""
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()

    @staticmethod
    def verify_log_file(log_path: str | Path) -> tuple[bool, str]:
        """Re-derive every hash and verify chain continuity on a completed JSONL file."""
        path = Path(log_path)
        if not path.exists():
            return False, f"File not found: {path}"

        lines: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError as err:
                    return False, f"Line {idx}: JSON parse error: {err}"

        if not lines:
            return False, "Log file is empty."

        for idx, line in enumerate(lines):
            stored_hash = line.get("hash")
            stored_prev = line.get("prev_hash")

            line_without_hash = {k: v for k, v in line.items() if k != "hash"}
            try:
                expected_hash = compute_sha256_hash(line_without_hash, stored_prev)
            except Exception as e:
                return False, f"Line {idx+1}: canonical serialization error: {e}"

            if stored_hash != expected_hash:
                return (
                    False,
                    f"Hash mismatch at step {line.get('step', idx+1)}: "
                    f"stored={stored_hash[:16]}... expected={expected_hash[:16]}...",
                )

            # Chain continuity check
            if idx > 0:
                expected_prev = lines[idx - 1].get("hash")
                if stored_prev != expected_prev:
                    return (
                        False,
                        f"Chain break at step {line.get('step', idx+1)}: "
                        f"prev_hash does not match previous line's hash.",
                    )

        return True, f"Hash chain VALID — {len(lines)} entries verified."
