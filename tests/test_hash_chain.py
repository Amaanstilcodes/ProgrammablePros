"""Tests for Cryptographic SHA-256 Hash Chain Integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.official_logger import OfficialLogger, canonical_json, compute_sha256_hash


def test_canonical_json_sorting_and_compact() -> None:
    data = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}
    result = canonical_json(data)
    assert result == '{"a":2,"m":{"a":4,"b":3},"z":1}'
    assert " " not in result


def test_canonical_json_forbids_nan() -> None:
    with pytest.raises(ValueError, match="forbids NaN"):
        canonical_json({"val": float("nan")})


def test_official_logger_hash_chain(tmp_path: Path) -> None:
    logger = OfficialLogger(base_dir=tmp_path, is_mock=True)
    log_file = logger.start_session("TEST-SESS-001", "snake")

    for step in range(1, 4):
        logger.log_move(
            step=step,
            state_before={"step": step - 1, "snake": [[0, 0]]},
            action="right",
            state_after={"step": step, "snake": [[1, 0]]},
            result="applied",
            decision={
                "questions": ["q1"],
                "probabilities": {"up": 0.1, "down": 0.1, "left": 0.1, "right": 0.7},
                "chosen": "right",
                "confidence": 0.7,
                "source": "laya",
            },
            timing={"decide_ms": 1, "api_ms": 2, "turn_ms": 5},
        )
    logger.close()

    is_valid, msg = OfficialLogger.verify_log_file(log_file)
    assert is_valid is True
    assert "3 entries verified" in msg


def test_official_logger_detects_tampering(tmp_path: Path) -> None:
    logger = OfficialLogger(base_dir=tmp_path, is_mock=True)
    log_file = logger.start_session("TEST-SESS-002", "snake")

    logger.log_move(
        step=1,
        state_before={"step": 0},
        action="right",
        state_after={"step": 1},
        result="applied",
        decision={
            "questions": ["q"],
            "probabilities": {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25},
            "chosen": "right",
            "confidence": 0.25,
            "source": "laya",
        },
        timing={"decide_ms": 1, "api_ms": 1, "turn_ms": 2},
    )
    logger.close()

    # Tamper with file
    content = log_file.read_text(encoding="utf-8")
    tampered = content.replace('"right"', '"left"', 1)
    log_file.write_text(tampered, encoding="utf-8")

    is_valid, msg = OfficialLogger.verify_log_file(log_file)
    assert is_valid is False
    assert "Hash mismatch" in msg or "Chain break" in msg

    # Also assert that tools/verify_log.py CLI rejects it
    from tools.verify_log import verify_log
    assert verify_log(str(log_file)) is False

