"""Main Game Loop Module.

Executes the autonomous bot game loop with 4 advanced strategies:
1. Feature 1: State Compression & Feature Engineering
2. Feature 2: Hypothesis-Driven "Typed Questions"
3. Feature 3: Expected Value (EV) Decision Making (Reward +10 / Penalty -5)
4. Feature 4: Failure-Adaptive Prompting (Short-Term Memory)

Usage:
    python main.py                  # original 1-D grid game (default)
    python main.py --game vault     # new 2-D Vault Runner heist game
    python main.py --game vault --no-ev  # argmax baseline (disables EV weighting)
"""

import argparse
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from laya_agent import LayaAgent
from logger import DecisionLogger
from mock_game import MockGame


# ── Game registry (one-line to add a third game) ─────────────────
def _build_vault():
    """Lazy import so vault deps don't affect classic game startup."""
    from vault_game import VaultRunnerGame, compress_vault_state
    return VaultRunnerGame, compress_vault_state


GAME_REGISTRY = {
    "classic": {
        "factory": lambda: (MockGame(start_pos=0, goal_pos=10, max_steps=30), None),
        "label": "1-D Grid Navigator",
        "goal_phrase": "the goal",
        "send_action_method": "send_action",  # MockGame uses send_action
        "is_done_attr": "is_game_over",       # MockGame uses attribute
    },
    "vault": {
        "factory": lambda: (_build_vault()[0](), _build_vault()[1]),
        "label": "Vault Runner (2-D Heist)",
        "goal_phrase": "the KEY then EXIT without being caught or hitting a hazard",
        "send_action_method": "step",         # VaultRunnerGame uses step
        "is_done_attr": None,                 # uses is_done() method
    },
}


def print_banner(game_label: str) -> None:
    """Print startup information to terminal."""
    print("=" * 70)
    print("[*] AI GAME BOT -- POWERED BY LAYA CALIBRATED PROBABILITIES [*]")
    print("=" * 70)
    print(f"  Game Mode:           {game_label}")
    print("  Decision Model:      Laya (Calibrated Probability Scoring)")
    print("  Active Strategies:   1. State Compression & Feature Engineering")
    print("                       2. Hypothesis-Driven Typed Questions")
    print("                       3. Expected Value (EV) Risk/Reward Optimization")
    print("                       4. Failure-Adaptive Short-Term Memory")
    print("  Log Output:          decision_log.jsonl")
    print("  Live Viewer:         Run 'streamlit run viewer.py' in a separate terminal")
    print("=" * 70 + "\n")


def _game_is_over(game: Any, reg: dict) -> bool:
    """Unified is-over check for both game types."""
    if reg["is_done_attr"] is not None:
        return getattr(game, reg["is_done_attr"])
    return game.is_done()


def run_bot(
    game_name: str = "classic",
    reward: float = 10.0,
    penalty: float = 5.0,
    use_ev: bool = True,
    silent: bool = False,
) -> Dict[str, Any]:
    """Execute autonomous game playing loop with EV decision-making and failure adaptation.

    Args:
        game_name: 'classic' or 'vault' (registry key).
        reward: Global EV payoff for actions leading toward goal progress (default: +10.0).
        penalty: Global EV penalty for actions failing to make progress (default: 5.0).
        use_ev: If False, select action by argmax(P) only -- EV baseline comparison.
        silent: Suppress print output (used for headless benchmark).

    Returns:
        Summary dict: steps, result, total_penalty (vault only).
    """
    reg = GAME_REGISTRY[game_name]

    if not silent:
        print_banner(reg["label"])

    # ── 1. Initialise components ──────────────────────────────────
    logger = DecisionLogger(log_path="decision_log.jsonl", clear_on_init=not silent)
    goal_phrase = reg["goal_phrase"]

    # Build game + optional compressor
    if game_name == "vault":
        from vault_game import VaultRunnerGame, compress_vault_state
        game = VaultRunnerGame()
        compressor = compress_vault_state
    else:
        game = MockGame(start_pos=0, goal_pos=10, max_steps=30)
        compressor = None

    agent = LayaAgent(use_mock=True, max_memory_turns=3)

    if not silent:
        print("[+] Initialising Game Session...")

    initial_state = game.get_state()

    if not silent:
        if game_name == "vault":
            print(f"   Grid: 8x8 | Steps: {initial_state['max_steps']} | Objective: KEY then EXIT\n")
            print(game.render_ascii())
            print()
        else:
            print(f"   Start Position: {initial_state['player_pos']} | Target Goal: {initial_state['goal_pos']}\n")

    step = 0
    while not _game_is_over(game, reg):
        step += 1
        try:
            # Step A: Read current game state
            current_state = game.get_state()
            available_actions = current_state.get("available_actions", [])

            if not available_actions:
                if not silent:
                    print(f"[Step {step}] [!] No available actions. Exiting.")
                break

            # Step B: Strategy 1 -- compress state; Strategy 2 -- formulate questions
            compressed: str
            if compressor is not None:
                compressed = compressor(current_state)
            else:
                compressed = agent.compress_state(current_state)

            probabilities: Dict[str, float] = agent.decide(
                current_state,
                available_actions,
                goal_phrase=goal_phrase,
                compressed_state_override=compressed,
            )
            questions_asked: Dict[str, str] = agent.get_last_questions()
            warnings_generated: Dict[str, str] = agent.get_last_warnings()

            # Step C: Strategy 3 -- Expected Value Decision Making
            payoffs: Optional[Dict[str, Tuple[float, float]]] = None
            if game_name == "vault" and use_ev:
                payoffs = {a: game.get_action_risk(current_state, a) for a in available_actions}

            expected_values: Dict[str, float] = agent.calculate_expected_values(
                probabilities=probabilities,
                reward=reward,
                penalty=penalty,
                payoffs=payoffs if use_ev else None,
            )

            # Select action: EV-argmax (default) or P-argmax (baseline)
            if use_ev:
                chosen_action: str = max(expected_values, key=expected_values.get)  # type: ignore[arg-type]
            else:
                chosen_action = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]

            chosen_ev: float = expected_values[chosen_action]
            chosen_warning: Optional[str] = warnings_generated.get(chosen_action)
            active_warning: Optional[str] = chosen_warning or (
                list(warnings_generated.values())[0] if warnings_generated else None
            )

            # Step D: Log decision to JSONL (Strategy 1-4 fields)
            extra_fields: Dict[str, Any] = {
                "game": game_name,
                "hidden_wall_discovered": current_state.get("hidden_wall_discovered"),
            }
            if payoffs:
                extra_fields["payoffs"] = {a: list(v) for a, v in payoffs.items()}

            log_state = dict(current_state)
            log_state.update(extra_fields)

            logger.log_decision(
                game_state=log_state,
                questions_asked=questions_asked,
                probabilities_returned=probabilities,
                chosen_action=chosen_action,
                compressed_state=compressed,
                expected_values=expected_values,
                expected_value=chosen_ev,
                adaptive_warning=active_warning,
            )

            # Step E: Console feedback
            if not silent:
                prob_str = ", ".join(f"'{a}': {p:.1%}" for a, p in probabilities.items())
                ev_str = ", ".join(f"'{a}': {ev:+.2f}" for a, ev in expected_values.items())
                warning_tag = f" | [ADAPTED: {active_warning[:40]}...]" if active_warning else ""

                if game_name == "vault":
                    px, py = current_state["player_pos"]
                    obj_dist = current_state["steps_remaining"]
                    print(
                        f"[Step {step:02d}] Pos:({px},{py}) Keys:{current_state['keys_held']} "
                        f"| Probs: [{prob_str}]\n"
                        f"          | EVs:  [{ev_str}]\n"
                        f"          | -> Move: '{chosen_action.upper()}' (EV: {chosen_ev:+.2f}){warning_tag}"
                    )
                    print(game.render_ascii())
                    print()
                else:
                    print(
                        f"[Step {step:02d}] Pos: {current_state['player_pos']:2d}/{current_state['goal_pos']} "
                        f"| Probs: [{prob_str}] "
                        f"| EVs: [{ev_str}] "
                        f"| -> Move: '{chosen_action.upper()}' (EV: {chosen_ev:+.2f}){warning_tag}"
                    )

            # Step F: Dispatch action
            if game_name == "vault":
                new_state = game.step(chosen_action)
                progress_made = new_state.get("progress_made", False)
                failure_detail = "Progress made" if progress_made else "No progress"
                hidden_hit = new_state.get("hidden_wall_discovered")
                if hidden_hit and not silent:
                    print(f"          [!] Hidden wall discovered at {hidden_hit}! WARNING injected next turn.")
            else:
                new_state = game.send_action(chosen_action)
                prev_dist = abs(current_state.get("goal_pos", 10) - current_state.get("player_pos", 0))
                new_dist = abs(new_state.get("goal_pos", 10) - new_state.get("player_pos", 0))
                progress_made = new_dist < prev_dist
                failure_detail = (
                    f"Progressed towards goal (dist {prev_dist} -> {new_dist})"
                    if progress_made
                    else f"No progress towards goal (dist {prev_dist} -> {new_dist})"
                )

            # Step G: Strategy 4 -- record outcome in failure-adaptive memory
            agent.record_outcome(
                action=chosen_action,
                prev_state=current_state,
                new_state=new_state,
                success=progress_made,
                details=failure_detail,
            )

            if not silent:
                time.sleep(0.5 if game_name == "vault" else 0.8)

        except KeyboardInterrupt:
            print("\n[!] Bot execution paused by user (KeyboardInterrupt).")
            sys.exit(0)
        except Exception as exc:
            if not silent:
                print(f"[Step {step}] [ERROR] {exc}")
            time.sleep(0.2)

    # ── Game completion ───────────────────────────────────────────
    final_state = game.get_state()
    result = final_state.get("status", "UNKNOWN")
    total_penalty = final_state.get("total_penalty", 0.0)

    if not silent:
        print("\n" + "=" * 70)
        if result == "WON":
            print(f"[WIN] SUCCESS! Reached objective in {final_state['current_step']} steps!")
        else:
            print(f"[OVER] Game Over with status: {result} at step {final_state['current_step']}.")
        if game_name == "vault":
            print(f"       Total penalty accumulated: {total_penalty:.1f}")
        print("[*] All decisions with compressed states & EVs recorded in 'decision_log.jsonl'.")
        print("=" * 70)

    return {
        "steps": final_state.get("current_step", step),
        "result": result,
        "total_penalty": total_penalty,
        "won": result == "WON",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laya AI Game Bot")
    parser.add_argument(
        "--game",
        choices=list(GAME_REGISTRY.keys()),
        default="classic",
        help="Game to run: 'classic' (1-D grid, default) or 'vault' (2-D heist).",
    )
    parser.add_argument(
        "--no-ev",
        action="store_true",
        default=False,
        help="Disable EV weighting and select actions by argmax(P) only (baseline).",
    )
    parser.add_argument("--reward", type=float, default=10.0, help="Global EV reward value.")
    parser.add_argument("--penalty", type=float, default=5.0, help="Global EV penalty value.")
    args = parser.parse_args()

    run_bot(
        game_name=args.game,
        reward=args.reward,
        penalty=args.penalty,
        use_ev=not args.no_ev,
    )
