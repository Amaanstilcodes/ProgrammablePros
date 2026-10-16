"""Tests for Compliance Layer and Invariants I1-I6."""

from __future__ import annotations

import pytest

from core.compliance_layer import ComplianceLayer


def test_normalize_probabilities_guarantees_sum() -> None:
    cl = ComplianceLayer()
    raw = {"up": 0.8, "down": 0.1, "left": 0.05, "right": 0.05}
    norm = cl.normalize_probabilities(raw)

    assert set(norm.keys()) == {"up", "down", "left", "right"}
    total = sum(norm.values())
    assert 0.98 <= total <= 1.02
    assert abs(total - 1.0) < 1e-4


def test_normalize_probabilities_handles_degenerate_zeros() -> None:
    cl = ComplianceLayer()
    raw = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
    norm = cl.normalize_probabilities(raw)
    assert norm == {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}


def test_snake_safety_fallback_blocks_180_reversal() -> None:
    cl = ComplianceLayer()
    state = {"direction": "right"}

    # Proposing reverse ('left') should trigger fallback
    fb = cl.check_snake_safety_fallback(state, "left")
    assert fb is not None
    assert fb in ("up", "down")

    # Proposing continuation ('right') or turns ('up', 'down') should not trigger fallback
    assert cl.check_snake_safety_fallback(state, "right") is None
    assert cl.check_snake_safety_fallback(state, "up") is None
    assert cl.check_snake_safety_fallback(state, "down") is None


def test_build_decision_block_invariants() -> None:
    cl = ComplianceLayer()
    norm = {"up": 0.1, "down": 0.2, "left": 0.3, "right": 0.4}
    block = cl.build_decision_block(
        questions=["q1"],
        normalized_probs=norm,
        chosen_action="right",
        source="laya",
    )

    assert block["chosen"] == "right"
    assert block["confidence"] == 0.4
    assert block["source"] == "laya"
    assert block["probabilities"] == norm
