"""Main Game Loop Module.

Executes the autonomous bot game loop with 4 advanced strategies:
1. Feature 1: State Compression & Feature Engineering
2. Feature 2: Hypothesis-Driven "Typed Questions"
3. Feature 3: Expected Value (EV) Decision Making (Reward +10 / Penalty -5)
4. Feature 4: Failure-Adaptive Prompting (Short-Term Memory)
"""

import sys
import time
from typing import Any, Dict, Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from laya_agent import LayaAgent
from logger import DecisionLogger
from mock_game import MockGame


def print_banner() -> None:
    """Print startup information to terminal."""
    print("=" * 70)
    print("[*] AI GAME BOT — POWERED BY LAYA CALIBRATED PROBABILITIES [*]")
    print("=" * 70)
    print("  Decision Model:      Laya (Calibrated Probability Scoring)")
    print("  Active Strategies:   1. State Compression & Feature Engineering")
    print("                       2. Hypothesis-Driven Typed Questions")
    print("                       3. Expected Value (EV) Risk/Reward Optimization")
    print("                       4. Failure-Adaptive Short-Term Memory")
    print("  Log Output:          decision_log.jsonl")
    print("  Live Viewer:         Run 'streamlit run viewer.py' in a separate terminal")
    print("=" * 70 + "\n")


def run_bot(reward: float = 10.0, penalty: float = 5.0) -> None:
    """Execute autonomous game playing loop with EV decision-making and failure adaptation.

    Args:
        reward: EV payoff for actions leading toward goal progress (default: +10.0).
        penalty: EV penalty for actions failing to make progress (default: 5.0).
    """
    print_banner()

    # 1. Initialize components
    # Set clear_on_init=True to start a clean log each run
    logger = DecisionLogger(log_path="decision_log.jsonl", clear_on_init=True)
    game = MockGame(start_pos=0, goal_pos=10, max_steps=30)
    agent = LayaAgent(use_mock=True, max_memory_turns=3)

    print("[+] Initializing Game Session...")
    initial_state = game.get_state()
    print(f"   Start Position: {initial_state['player_pos']} | Target Goal: {initial_state['goal_pos']}\n")

    step = 0
    while not game.is_game_over:
        step += 1
        try:
            # Step A: Read current game state
            current_state = game.get_state()
            available_actions = current_state.get("available_actions", [])

            if not available_actions:
                print(f"[Step {step}] [!] No available actions in state. Exiting.")
                break

            # Step B: Query Laya for calibrated probabilities (Features 1, 2, 4)
            probabilities: Dict[str, float] = agent.decide(current_state, available_actions)
            questions_asked: Dict[str, str] = agent.get_last_questions()
            compressed_state: str = agent.get_last_compressed_state()
            warnings_generated: Dict[str, str] = agent.get_last_warnings()

            # Step C: Expected Value (EV) Decision Making (Feature 3)
            # EV = (P_success * Reward) - (P_failure * Penalty)
            expected_values: Dict[str, float] = agent.calculate_expected_values(
                probabilities=probabilities,
                reward=reward,
                penalty=penalty,
            )

            # Select the action with the highest Expected Value
            chosen_action: str = max(expected_values, key=expected_values.get)  # type: ignore[arg-type]
            chosen_ev: float = expected_values[chosen_action]
            chosen_warning: Optional[str] = warnings_generated.get(chosen_action)

            # If any failure warning is active in this turn, capture it for logging telemetry
            active_warning: Optional[str] = chosen_warning or (
                list(warnings_generated.values())[0] if warnings_generated else None
            )

            # Step D: Log decision with all 4 feature metrics to JSONL
            logger.log_decision(
                game_state=current_state,
                questions_asked=questions_asked,
                probabilities_returned=probabilities,
                chosen_action=chosen_action,
                compressed_state=compressed_state,
                expected_values=expected_values,
                expected_value=chosen_ev,
                adaptive_warning=active_warning,
            )

            # Step E: Visual terminal feedback
            prob_str = ", ".join(f"'{act}': {prob:.1%}" for act, prob in probabilities.items())
            ev_str = ", ".join(f"'{act}': {ev:+.2f}" for act, ev in expected_values.items())
            warning_tag = f" | [⚠️ ADAPTED: {active_warning}]" if active_warning else ""

            print(
                f"[Step {step:02d}] Pos: {current_state['player_pos']:2d}/{current_state['goal_pos']} "
                f"| Probs: [{prob_str}] "
                f"| EVs: [{ev_str}] "
                f"| -> Move: '{chosen_action.upper()}' (EV: {chosen_ev:+.2f}){warning_tag}"
            )

            # Step F: Dispatch action to game environment
            new_state = game.send_action(chosen_action)

            # Step G: Evaluate outcome & record failure/success memory (Feature 4)
            prev_dist = abs(current_state.get("goal_pos", 10) - current_state.get("player_pos", 0))
            new_dist = abs(new_state.get("goal_pos", 10) - new_state.get("player_pos", 0))

            # Action succeeded if distance to goal decreased; failed if stayed same or increased
            move_succeeded = new_dist < prev_dist
            failure_detail = (
                f"Progressed towards goal (dist {prev_dist} -> {new_dist})"
                if move_succeeded
                else f"No progress towards goal (dist {prev_dist} -> {new_dist})"
            )

            agent.record_outcome(
                action=chosen_action,
                prev_state=current_state,
                new_state=new_state,
                success=move_succeeded,
                details=failure_detail,
            )

            # Pacing delay so live Streamlit viewer can visually display updates
            time.sleep(0.8)

        except KeyboardInterrupt:
            print("\n[!] Bot execution paused by user (KeyboardInterrupt).")
            sys.exit(0)
        except Exception as exc:
            print(f"[Step {step}] [ERROR] Error during decision cycle: {exc}")
            time.sleep(1.0)

    # Game Completion
    final_state = game.get_state()
    print("\n" + "=" * 70)
    if final_state.get("status") == "WON":
        print(f"[WIN] SUCCESS! Player reached goal {final_state['goal_pos']} in {final_state['current_step']} steps!")
    else:
        print(f"[OVER] Game Over with status: {final_state.get('status')} at step {final_state['current_step']}.")
    print("[*] All decisions with compressed states & EVs recorded in 'decision_log.jsonl'.")
    print("=" * 70)


if __name__ == "__main__":
    run_bot()
