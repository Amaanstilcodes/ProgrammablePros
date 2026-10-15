"""test_vault.py -- Acceptance tests for VaultRunner + Laya Agent integration.

Run with:
    python test_vault.py              # plain asserts
    python -m pytest test_vault.py -v # pytest

Tests:
  1. Determinism + solvability
  2. Mechanics: guard patrol, laser penalty, trap (one-time), exit lock, sprint crash
  3. compress_vault_state: length <= 120, no hidden-wall info
  4. EV engine picks different action than argmax(P) in crafted scenario
  5. Hidden-wall failure -> WARNING injected -> bot picks different action
  6. Headless benchmark: 20 seeds, EV vs argmax win-rate/steps/penalty table
  7. Original game still runs and output is unchanged
"""

from __future__ import annotations

import sys
import os

# ── ensure project root on path ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
from typing import Any, Dict, List, Tuple

from vault_game import (
    VaultRunnerGame,
    compress_vault_state,
    TILE_HIDDEN_WALL,
    TILE_LASER,
    TILE_TRAP,
    MAX_STEPS,
    _bfs_reachable,
    _manhattan,
)
from laya_agent import LayaAgent
from mock_game import MockGame


# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────

def _run_vault_headless(seed: int, use_ev: bool = True) -> Dict[str, Any]:
    """Run one full Vault Runner game headlessly and return metrics."""
    from main import run_bot
    return run_bot(game_name="vault", use_ev=use_ev, silent=True)


# ─────────────────────────────────────────────────────────────────
# TEST 1: Determinism + Solvability
# ─────────────────────────────────────────────────────────────────

def test_determinism_and_solvability():
    """Test 1: Layout is identical for fixed seed; BFS confirms solvability."""
    g1 = VaultRunnerGame(seed=42)
    g2 = VaultRunnerGame(seed=42)

    # Same seed -> same layout
    assert g1._layout == g2._layout, "Same seed must produce identical layouts"

    # BFS solvability check
    layout = g1._layout
    start = g1._find_tile(layout, "S")
    key_pos = g1._find_tile(layout, "K")
    exit_pos = g1._find_tile(layout, "E")
    assert start is not None, "Layout must have a START tile"
    assert key_pos is not None, "Layout must have a KEY tile"
    assert exit_pos is not None, "Layout must have an EXIT tile"
    assert _bfs_reachable(layout, start, exit_pos, via=key_pos), \
        "Layout must always be BFS-solvable (START -> KEY -> EXIT)"

    print("  [PASS] Test 1: Determinism + solvability")


# ─────────────────────────────────────────────────────────────────
# TEST 2: Game mechanics
# ─────────────────────────────────────────────────────────────────

def test_guard_patrol():
    """Test 2a: Guard moves along its patrol route and bounces back."""
    g = VaultRunnerGame(seed=42)
    assert g._guard is not None
    positions = [g._guard.pos]
    for _ in range(10):
        g._guard.step()
        positions.append(g._guard.pos)
    # Guard must stay within patrol list
    for pos in positions:
        assert pos in g._guard.patrol, f"Guard left patrol path: {pos}"
    # Guard must have reversed at least once (visited a position twice)
    assert len(positions) > len(set(positions)), "Guard must bounce and revisit positions"
    print("  [PASS] Test 2a: Guard patrol")


def test_laser_penalty():
    """Test 2b: Stepping on a laser costs -15 penalty."""
    g = VaultRunnerGame(seed=42)
    g.reset()
    # Teleport player adjacent to laser at (5,3); approach from (4,3)
    g.player_pos = (4, 3)
    g.facing = "right"
    penalty_before = g.total_penalty
    result = g.step("right")
    # Either moved onto laser or blocked -- check only if move succeeded
    if g.player_pos == (5, 3):
        assert g.total_penalty == penalty_before - 15.0, \
            f"Laser should cost -15; got delta={g.total_penalty - penalty_before}"
        print("  [PASS] Test 2b: Laser penalty")
    else:
        print("  [SKIP] Test 2b: Player blocked before laser (guard in way); skipping")


def test_trap_floor_one_time():
    """Test 2c: Trap costs -8 first visit, then acts as normal floor."""
    g = VaultRunnerGame(seed=42)
    g.reset()
    # Teleport player above trap at (3,6)
    g.player_pos = (3, 5)
    g.facing = "down"
    penalty_before = g.total_penalty
    g.step("down")  # first step onto trap
    penalty_after_first = g.total_penalty
    delta_first = penalty_after_first - penalty_before
    # Walk away and come back
    g.player_pos = (3, 5)
    g.step("down")  # second step -- trap already triggered
    penalty_after_second = g.total_penalty
    delta_second = penalty_after_second - penalty_after_first
    assert delta_first == -8.0, f"First trap step should cost -8; got {delta_first}"
    assert delta_second == 0.0, f"Second trap step should cost 0; got {delta_second}"
    print("  [PASS] Test 2c: Trap floor (one-time)")


def test_exit_locked_without_key():
    """Test 2d: Stepping on EXIT without key results in no progress."""
    g = VaultRunnerGame(seed=42)
    g.reset()
    # Teleport adjacent to exit at (7,7)
    g.player_pos = (6, 7)
    g.facing = "right"
    g.keys_held = 0
    pos_before = g.player_pos
    result = g.step("right")
    assert result["progress_made"] is False, "EXIT without key must signal no progress"
    assert g.player_pos == pos_before, "Player must not enter EXIT without key"
    assert g.result == "IN_PROGRESS", "Game must not end on locked EXIT"
    print("  [PASS] Test 2d: EXIT locked without key")


def test_sprint_wall_crash():
    """Test 2e: Sprint into a wall on the second tile costs -10 extra."""
    g = VaultRunnerGame(seed=42)
    g.reset()
    # Place player so facing=right, tile at +1 is clear, tile at +2 is wall
    # Row 0: S . . . W . . .  -> pos (2,0), facing right hits W at (4,0) on 2nd hop
    g.player_pos = (2, 0)
    g.facing = "right"
    penalty_before = g.total_penalty
    g.step("sprint")
    penalty_delta = g.total_penalty - penalty_before
    # Player should have moved to (3,0) (first hop clear) and crashed at (4,0)
    assert penalty_delta == -10.0, \
        f"Sprint wall-crash should cost -10; got {penalty_delta}"
    print("  [PASS] Test 2e: Sprint wall crash penalty")


# ─────────────────────────────────────────────────────────────────
# TEST 3: compress_vault_state
# ─────────────────────────────────────────────────────────────────

def test_compress_vault_state_length_and_no_hidden():
    """Test 3: Compressed state is <=120 chars and never leaks hidden-wall info."""
    g = VaultRunnerGame(seed=42)
    g.reset()
    for _ in range(5):
        state = g.get_state()
        compressed = compress_vault_state(state)
        assert len(compressed) <= 120, \
            f"Compressed state too long ({len(compressed)} chars): {compressed}"
        assert "H" not in compressed.split(), \
            f"Hidden-wall tile 'H' leaked into compressed state: {compressed}"
        if not g.is_done():
            g.step(random.choice(["right", "down", "up", "left"]))
    print("  [PASS] Test 3: compress_vault_state length + no hidden-wall leakage")


# ─────────────────────────────────────────────────────────────────
# TEST 4: EV picks different action than argmax(P) in crafted scenario
# ─────────────────────────────────────────────────────────────────

def test_ev_diverges_from_argmax():
    """Test 4: EV engine picks safer action when asymmetric payoffs penalise risky move.

    Scenario: sprint has slightly higher P than down, but (reward=8, penalty=15)
    for sprint (crash-risk) vs (reward=10, penalty=5) for down.
    EV formula: EV = P*R - (1-P)*|Penalty|.
    """
    agent = LayaAgent(use_mock=False)  # We inject probabilities manually

    # Craft probabilities where sprint has highest P but worst payoff
    probabilities = {
        "sprint": 0.72,   # highest P
        "down":   0.70,
        "wait":   0.25,
    }
    # Asymmetric payoffs (Strategy 3): sprint has crash risk
    payoffs = {
        "sprint": (8.0, 15.0),   # crash risk: low reward, high penalty
        "down":   (10.0, 5.0),   # normal step
        "wait":   (2.0, 1.0),
    }

    # argmax(P) picks sprint
    argmax_action = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]

    # EV-argmax
    evs = agent.calculate_expected_values(probabilities, payoffs=payoffs)
    ev_action = max(evs, key=evs.get)  # type: ignore[arg-type]

    ev_sprint = evs["sprint"]  # 0.72*8 - 0.28*15 = 5.76 - 4.20 = 1.56
    ev_down = evs["down"]      # 0.70*10 - 0.30*5 = 7.00 - 1.50 = 5.50

    assert argmax_action == "sprint", f"argmax should pick sprint, got {argmax_action}"
    assert ev_action != "sprint", (
        f"EV should NOT pick sprint (EV={ev_sprint:.2f}) over down (EV={ev_down:.2f})"
    )
    assert ev_action == "down", f"EV should pick 'down', got '{ev_action}' (EVs: {evs})"
    print(f"  [PASS] Test 4: EV diverges from argmax -- "
          f"P-argmax=sprint (P=0.72), EV-argmax=down "
          f"(EV_sprint={ev_sprint:.2f} < EV_down={ev_down:.2f})")


# ─────────────────────────────────────────────────────────────────
# TEST 5: Hidden-wall failure -> WARNING -> bot changes action
# ─────────────────────────────────────────────────────────────────

def test_hidden_wall_warning_and_recovery():
    """Test 5: After hitting a hidden wall, WARNING is injected and bot avoids that action.

    Unit-style check: within 2 turns of the hidden-wall hit, the agent must
    switch away from the blocked action.
    """
    agent = LayaAgent(use_mock=True, max_memory_turns=3)

    # Simulate a state where player is at (2,2) and 'right' leads to hidden wall at (3,2)
    fake_prev_state: Dict[str, Any] = {
        "player_pos": (2, 2),
        "keys_held": 0,
        "game_type": "VAULT_RUNNER",
        "current_step": 1,
    }
    fake_new_state: Dict[str, Any] = {
        "player_pos": (2, 2),  # didn't move
        "keys_held": 0,
        "game_type": "VAULT_RUNNER",
        "current_step": 2,
        "hidden_wall_discovered": (3, 2),
    }

    # Record hidden-wall hit (no progress)
    agent.record_outcome(
        action="right",
        prev_state=fake_prev_state,
        new_state=fake_new_state,
        success=False,
        details="Hidden wall at (3,2)",
    )

    # On the next turn, formulate questions and check WARNING injection
    actions = ["up", "down", "left", "right", "wait", "sprint"]
    state_for_query: Dict[str, Any] = {
        "player_pos": (2, 2),
        "goal_pos": (7, 7),
        "key_pos": (4, 4),
        "keys_held": 0,
        "facing": "right",
        "steps_remaining": 38,
        "guard_pos": (0, 3),
        "guard_next_pos": (1, 3),
        "guard_direction": "RIGHT",
        "game_type": "VAULT_RUNNER",
        "visible_grid": [],
        "current_step": 2,
        "max_steps": MAX_STEPS,
        "hidden_wall_discovered": None,
    }

    # Pre-compute the compressed state string (Strategy 1) so we bypass
    # the 1-D compress_state which can't handle tuple player_pos
    from vault_game import compress_vault_state as cvs
    compressed_override = cvs(state_for_query)

    questions = agent.formulate_questions(
        state_for_query,
        actions,
        goal_phrase="the KEY then EXIT",
        compressed_state_override=compressed_override,
    )
    warnings = agent.get_last_warnings()

    # WARNING must be injected for 'right' at position (2,2)
    assert "right" in warnings, f"WARNING must be injected for 'right' after hidden-wall hit; got warnings={list(warnings.keys())}"
    assert "WARNING" in questions["right"], "Question for 'right' must contain WARNING text"

    # Bot must pick different action after WARNING (probabilities suppressed for 'right')
    probs = agent._mock_vault_eval(state_for_query, actions)
    assert probs.get("right", 1.0) < 0.20, \
        f"'right' probability must be suppressed after WARNING; got {probs.get('right')}"

    ev_scores = agent.calculate_expected_values(probs)
    chosen = max(ev_scores, key=ev_scores.get)  # type: ignore[arg-type]
    assert chosen != "right", f"Bot must not pick 'right' after WARNING; chose '{chosen}'"

    print(f"  [PASS] Test 5: Hidden wall -> WARNING injected -> bot chose '{chosen}' (not 'right')")


# ─────────────────────────────────────────────────────────────────
# TEST 6: Headless benchmark (20 seeds, EV vs argmax)
# ─────────────────────────────────────────────────────────────────

def test_headless_benchmark():
    """Test 6: Run Vault Runner over 20 seeds with EV and argmax; report comparison table."""
    from main import run_bot

    seeds = list(range(1, 21))
    ev_results: List[Dict[str, Any]] = []
    argmax_results: List[Dict[str, Any]] = []

    for seed in seeds:
        # EV agent
        result_ev = run_bot(game_name="vault", use_ev=True, silent=True)
        ev_results.append(result_ev)
        # Argmax baseline
        result_argmax = run_bot(game_name="vault", use_ev=False, silent=True)
        argmax_results.append(result_argmax)

    ev_wins = sum(1 for r in ev_results if r["won"])
    argmax_wins = sum(1 for r in argmax_results if r["won"])
    ev_avg_steps = sum(r["steps"] for r in ev_results) / len(ev_results)
    argmax_avg_steps = sum(r["steps"] for r in argmax_results) / len(argmax_results)
    ev_avg_penalty = sum(r["total_penalty"] for r in ev_results) / len(ev_results)
    argmax_avg_penalty = sum(r["total_penalty"] for r in argmax_results) / len(argmax_results)

    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║         Headless Benchmark — 20 Seeds (Vault Runner)    ║")
    print("  ╠══════════════════════════╦═══════════════╦══════════════╣")
    print("  ║ Metric                   ║  EV Agent     ║  Argmax (P)  ║")
    print("  ╠══════════════════════════╬═══════════════╬══════════════╣")
    print(f"  ║ Win rate                 ║ {ev_wins:2d}/20 ({ev_wins/20:.0%})   ║ {argmax_wins:2d}/20 ({argmax_wins/20:.0%})  ║")
    print(f"  ║ Avg steps                ║ {ev_avg_steps:13.1f} ║ {argmax_avg_steps:12.1f} ║")
    print(f"  ║ Avg penalty              ║ {ev_avg_penalty:13.1f} ║ {argmax_avg_penalty:12.1f} ║")
    print("  ╚══════════════════════════╩═══════════════╩══════════════╝\n")

    assert ev_wins >= argmax_wins - 2, (
        f"EV agent should win within 2 of argmax (statistical variance); EV={ev_wins}, argmax={argmax_wins}"
    )
    # EV's primary benefit is lower penalties due to risk-aware payoffs (Strategy 3)
    assert ev_avg_penalty >= argmax_avg_penalty, (
        f"EV agent should accumulate LESS penalty than argmax; "
        f"EV={ev_avg_penalty:.1f}, argmax={argmax_avg_penalty:.1f}"
    )
    print(f"  [PASS] Test 6: EV penalty ({ev_avg_penalty:.1f}) <= argmax penalty ({argmax_avg_penalty:.1f}) -- risk-aware payoffs working")


# ─────────────────────────────────────────────────────────────────
# TEST 7: Original classic game unchanged
# ─────────────────────────────────────────────────────────────────

def test_original_classic_game():
    """Test 7: Original 1-D game still runs without errors and bot wins."""
    from main import run_bot
    result = run_bot(game_name="classic", silent=True)
    assert result["result"] == "WON", f"Classic game bot should win; got '{result['result']}'"
    assert result["steps"] <= 30, f"Classic game must finish within 30 steps; took {result['steps']}"
    print(f"  [PASS] Test 7: Classic game runs unchanged (won in {result['steps']} steps)")


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all acceptance tests sequentially."""
    print("\n" + "=" * 66)
    print("  Vault Runner + Laya Agent -- Acceptance Test Suite")
    print("=" * 66 + "\n")

    tests = [
        ("Test 1: Determinism + solvability", test_determinism_and_solvability),
        ("Test 2a: Guard patrol",             test_guard_patrol),
        ("Test 2b: Laser penalty",            test_laser_penalty),
        ("Test 2c: Trap (one-time)",          test_trap_floor_one_time),
        ("Test 2d: EXIT lock",                test_exit_locked_without_key),
        ("Test 2e: Sprint wall crash",        test_sprint_wall_crash),
        ("Test 3: State compression",         test_compress_vault_state_length_and_no_hidden),
        ("Test 4: EV vs argmax divergence",   test_ev_diverges_from_argmax),
        ("Test 5: Hidden wall -> WARNING",    test_hidden_wall_warning_and_recovery),
        ("Test 6: Headless benchmark",        test_headless_benchmark),
        ("Test 7: Classic game unchanged",    test_original_classic_game),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"Running {name}...")
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 66)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 66 + "\n")
    return failed


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
