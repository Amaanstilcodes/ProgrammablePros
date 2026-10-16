"""Tests for Fallback Budget Enforcement (PDF §4 <= 10%)."""

from __future__ import annotations

from core.compliance_layer import ComplianceLayer


def test_fallback_budget_within_limit() -> None:
    cl = ComplianceLayer()
    # 5 fallbacks out of 100 accepted = 5% <= 10%
    is_valid, ratio = cl.verify_fallback_budget(fallback_count=5, accepted_count=100)
    assert is_valid is True
    assert ratio == 0.05


def test_fallback_budget_exceeded() -> None:
    cl = ComplianceLayer()
    # 12 fallbacks out of 100 accepted = 12% > 10%
    is_valid, ratio = cl.verify_fallback_budget(fallback_count=12, accepted_count=100)
    assert is_valid is False
    assert ratio == 0.12


def test_fallback_budget_zero_accepted() -> None:
    cl = ComplianceLayer()
    is_valid, ratio = cl.verify_fallback_budget(fallback_count=0, accepted_count=0)
    assert is_valid is True
    assert ratio == 0.0
