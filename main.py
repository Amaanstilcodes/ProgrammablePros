"""Main Game Loop Module.

Executes the autonomous bot game loop:
1. Observes game state (Mock / Live API)
2. Asks Laya for calibrated action probabilities
3. Picks the highest-probability action
4. Logs the structured decision to JSONL
5. Dispatches action to game environment
"""

import sys
import time
from typing import Any, Dict

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
    print("=" * 65)
    print("[*] AI GAME BOT (POWERED BY LAYA PROBABILITY CALIBRATION) [*]")
    print("=" * 65)
    print("  Decision Model: Laya (Calibrated Probability Scoring)")
    print("  Log Output:     decision_log.jsonl")
    print("  Live Viewer:    Run 'streamlit run viewer.py' in a separate terminal")
    print("=" * 65 + "\n")


def run_bot() -> None:
    """Execute autonomous game playing loop."""
    print_banner()

    # 1. Initialize components
    # Set clear_on_init=True if you wish to start a clean log each run, or False to append
    logger = DecisionLogger(log_path="decision_log.jsonl", clear_on_init=True)
    game = MockGame(start_pos=0, goal_pos=10, max_steps=30)
    agent = LayaAgent(use_mock=True)

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

            # Step B: Query Laya for calibrated probabilities
            probabilities: Dict[str, float] = agent.decide(current_state, available_actions)
            questions_asked: Dict[str, str] = agent.get_last_questions()

            # Step C: Select optimal action (highest calibrated probability)
            chosen_action: str = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]

            # Step D: Log decision to JSONL
            logger.log_decision(
                game_state=current_state,
                questions_asked=questions_asked,
                probabilities_returned=probabilities,
                chosen_action=chosen_action,
            )

            # Step E: Visual terminal feedback
            prob_str = ", ".join(f"'{act}': {prob:.1%}" for act, prob in probabilities.items())
            print(
                f"[Step {step:02d}] Pos: {current_state['player_pos']:2d}/{current_state['goal_pos']} "
                f"| Probabilities: [{prob_str}] "
                f"| -> Move: '{chosen_action.upper()}'"
            )

            # Step F: Dispatch action to game environment
            new_state = game.send_action(chosen_action)

            # Pacing delay so live Streamlit viewer can visually display updates
            time.sleep(0.8)

        except KeyboardInterrupt:
            print("\n[!] Bot execution paused by user (KeyboardInterrupt).")
            sys.exit(0)
        except Exception as exc:
            print(f"[Step {step}] [ERROR] Error during decision cycle: {exc}")
            # Prevent rapid failure spin-loop
            time.sleep(1.0)

    # Game Completion
    final_state = game.get_state()
    print("\n" + "=" * 65)
    if final_state.get("status") == "WON":
        print(f"[WIN] SUCCESS! Player reached goal {final_state['goal_pos']} in {final_state['current_step']} steps!")
    else:
        print(f"[OVER] Game Over with status: {final_state.get('status')} at step {final_state['current_step']}.")
    print("[*] All decisions recorded in 'decision_log.jsonl'.")
    print("=" * 65)


if __name__ == "__main__":
    run_bot()
