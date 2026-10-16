"""Tests for StepLedger and Rejection Numbering Policies."""

from __future__ import annotations

from core.step_ledger import StepLedger


def test_step_ledger_policy_a_repeat_last_applied() -> None:
    ledger = StepLedger(policy="repeat_last_applied", max_accepted_steps=300)

    # Move 1: Applied -> step 1
    assert ledger.record_move("applied") == 1
    assert ledger.accepted_count == 1
    assert ledger.rejected_count == 0

    # Move 2: Applied -> step 2
    assert ledger.record_move("applied") == 2
    assert ledger.accepted_count == 2

    # Move 3: Rejected -> repeats step 2, does NOT advance accepted
    assert ledger.record_move("rejected") == 2
    assert ledger.accepted_count == 2
    assert ledger.rejected_count == 1

    # Move 4: Applied -> advances to step 3 (unbroken applied sequence 1, 2, 3)
    assert ledger.record_move("applied") == 3
    assert ledger.accepted_count == 3


def test_step_ledger_policy_b_pending_ordinal() -> None:
    ledger = StepLedger(policy="pending_ordinal", max_accepted_steps=300)

    # Move 1: Applied -> step 1
    assert ledger.record_move("applied") == 1

    # Move 2: Rejected -> uses pending next_step (2), does not advance
    assert ledger.record_move("rejected") == 2
    assert ledger.accepted_count == 1
    assert ledger.rejected_count == 1

    # Move 3: Applied -> step 2
    assert ledger.record_move("applied") == 2
    assert ledger.accepted_count == 2


def test_step_ledger_cap_enforcement() -> None:
    ledger = StepLedger(policy="repeat_last_applied", max_accepted_steps=2)
    assert not ledger.is_cap_reached

    ledger.record_move("applied")
    ledger.record_move("rejected")  # does not count toward cap
    assert not ledger.is_cap_reached

    ledger.record_move("applied")
    assert ledger.is_cap_reached
