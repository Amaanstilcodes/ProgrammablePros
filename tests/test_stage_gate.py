"""Tests for StageGate (Rule R8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.stage_gate import StageGate, StageLockedError


def test_stage_gate_blocks_2048_by_default(tmp_path: Path) -> None:
    gate = StageGate(log_root=tmp_path / "logs", unlock_marker_dir=tmp_path / "submission")
    unlocked, msg = gate.check_stage2_unlocked()
    assert unlocked is False
    assert "STAGE 2 LOCKED" in msg

    with pytest.raises(StageLockedError):
        gate.assert_stage_unlocked("2048")


def test_stage_gate_allows_override(tmp_path: Path) -> None:
    gate = StageGate(log_root=tmp_path / "logs", unlock_marker_dir=tmp_path / "submission")
    unlocked, msg = gate.check_stage2_unlocked(allow_override=True)
    assert unlocked is True
    assert "Override" in msg


def test_stage_gate_allows_when_token_present(tmp_path: Path) -> None:
    unlock_dir = tmp_path / "submission" / "verified"
    unlock_dir.mkdir(parents=True)
    token_file = unlock_dir / "stage1_unlock.json"
    token_file.write_text('{"organizer_token": "CONFIRMED_EVA02_UNLOCKED"}', encoding="utf-8")

    gate = StageGate(log_root=tmp_path / "logs", unlock_marker_dir=unlock_dir)
    unlocked, msg = gate.check_stage2_unlocked()
    assert unlocked is True
    assert "CONFIRMED_EVA02_UNLOCKED" in msg
