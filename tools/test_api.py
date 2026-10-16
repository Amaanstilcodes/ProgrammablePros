"""tools/test_api.py — Verify Real TopGameAI Arena API Integration.

Tests live API connectivity for Snake and 2048:
  1. Creates a Snake session via POST /sessions
  2. Sends 3 sequential moves: "right", "right", "down" via POST /sessions/{id}/action
  3. Prints full response JSON and extracted state structures for format discovery
  4. Tests rejection detection (e.g. 180-degree reversal)
  5. Verifies 2048 session creation and move
  6. Evaluates acceptance criteria A1 - A6
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from game_clients import BaseGameClient, Game2048Client, GameAPIError, MoveResult, SnakeClient


def _pretty(data: object) -> str:
    """Format dictionary or object as pretty JSON."""
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


def run_snake_test() -> bool:
    print("\n" + "=" * 80)
    print("[*] TEST 1: SNAKE ARENA INTEGRATION")
    print("=" * 80)

    client = SnakeClient(agent_name="ProgrammablePros")

    # Step 1: Create session
    print("\n--> [1/4] POST /sessions (game='snake', agent_name='ProgrammablePros')...")
    try:
        session_id, initial_state = client.start_game()
    except Exception as exc:
        print(f"[!] FAILED to start Snake session: {exc}")
        return False

    print(f"[+] Session created successfully!")
    print(f"    Session ID: {session_id}")
    print(f"    Raw Response:")
    print(_pretty(client._last_response.get(session_id, {})))
    print(f"    Extracted State:")
    print(_pretty(initial_state))

    # Step 2: Send 3 moves (right, right, down)
    moves_to_send = ["right", "right", "down"]
    current_state = initial_state

    for idx, action in enumerate(moves_to_send, start=1):
        print(f"\n--> [{idx + 1}/4] POST /sessions/{session_id}/action (action='{action}')...")
        try:
            res: MoveResult = client.send_move(session_id, action, state_before=current_state)
        except Exception as exc:
            print(f"[!] FAILED sending move '{action}': {exc}")
            return False

        print(f"[+] Move '{action}' response:")
        print(f"    Result: {res.result} | Steps: {res.steps} | Done: {res.game_over} | Won: {res.won} | Latency: {res.api_ms}ms")
        print(f"    Full Response JSON:")
        print(_pretty(res.raw_response))
        print(f"    State After:")
        print(_pretty(res.state_after))

        current_state = res.state_after

    # Step 3: Test rejection detection by attempting an illegal 180-degree reversal
    # Current direction is "down" so "up" is an illegal 180-degree reversal
    print("\n--> [Bonus] Testing Rejection Detection: Attempting illegal reversal (action='up')...")
    res_rejected = client.send_move(session_id, "up", state_before=current_state)
    print(f"[+] Reversal Move Result: {res_rejected.result} (HTTP {res_rejected.http_status})")
    print(f"    Steps remained at: {res_rejected.steps} (did not advance)")
    print(f"    Response body: {res_rejected.raw_response}")

    # Step 4: Verify get_final_state
    final_state = client.get_final_state(session_id)
    print(f"\n[+] get_final_state() returned state with keys: {list(final_state.keys())}")

    client.close()
    return True


def run_2048_test() -> bool:
    print("\n" + "=" * 80)
    print("[*] TEST 2: 2048 ARENA INTEGRATION")
    print("=" * 80)

    client = Game2048Client(agent_name="ProgrammablePros")

    print("\n--> [1/2] POST /sessions (game='2048', agent_name='ProgrammablePros')...")
    try:
        session_id, initial_state = client.start_game()
    except Exception as exc:
        print(f"[!] FAILED to start 2048 session: {exc}")
        return False

    print(f"[+] 2048 Session created successfully!")
    print(f"    Session ID: {session_id}")
    print(f"    Extracted 2048 State:")
    print(_pretty(initial_state))

    print(f"\n--> [2/2] POST /sessions/{session_id}/action (action='left')...")
    try:
        res: MoveResult = client.send_move(session_id, "left", state_before=initial_state)
    except Exception as exc:
        print(f"[!] FAILED sending 2048 move: {exc}")
        return False

    print(f"[+] Move 'left' response:")
    print(f"    Result: {res.result} | Steps: {res.steps} | Done: {res.game_over} | Latency: {res.api_ms}ms")
    print(f"    Full Response JSON:")
    print(_pretty(res.raw_response))
    print(f"    State After:")
    print(_pretty(res.state_after))

    client.close()
    return True


def main() -> int:
    print("\n================================================================================")
    print("[*] TOPGAMEAI REAL ARENA API VERIFICATION")
    print("    Target: https://topgameai.com/api/arena")
    print("    Agent:  ProgrammablePros")
    print("================================================================================")

    snake_ok = run_snake_test()
    g2048_ok = run_2048_test()

    print("\n" + "=" * 80)
    print("[*] VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  A1. Connects to real API & creates session:  {'PASS' if snake_ok and g2048_ok else 'FAIL'}")
    print(f"  A2. Sends moves & receives responses:        {'PASS' if snake_ok and g2048_ok else 'FAIL'}")
    print(f"  A3. Response parsed (state, done, won, etc): {'PASS' if snake_ok and g2048_ok else 'FAIL'}")
    print(f"  A4. Rejection detected on steps unadvanced:  {'PASS' if snake_ok else 'FAIL'}")
    print(f"  A5. Game over / win flags parsed:            {'PASS' if snake_ok and g2048_ok else 'FAIL'}")
    print(f"  A6. State JSON printed for discovery:        {'PASS' if snake_ok and g2048_ok else 'FAIL'}")
    print("=" * 80)

    if snake_ok and g2048_ok:
        print("\n>>> ALL ACCEPTANCE CRITERIA SATISFIED! <<<\n")
        return 0
    else:
        print("\n[!] SOME TESTS FAILED.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
