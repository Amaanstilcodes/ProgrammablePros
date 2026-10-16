"""Log Verification Tool — Genesis 2.0 Track EVA 02.

CLI tool to verify cryptographic SHA-256 hash chains and probability invariants
of an official JSONL log file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.official_logger import OfficialLogger


def verify(log_file: str) -> bool:
    print(f"[*] Auditing log file: {log_file}")
    is_valid, msg = OfficialLogger.verify_log_file(log_file)
    if not is_valid:
        print(f"[FAIL] Hash Chain Invalid: {msg}")
        return False

    print(f"[OK] {msg}")

    # Verify probability invariants
    with open(log_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            entry = json.loads(line_str)
            decision = entry.get("decision", {})
            probs = decision.get("probabilities", {})

            if set(probs.keys()) != {"up", "down", "left", "right"}:
                print(f"[FAIL] Step {entry.get('step', idx)}: Missing direction keys: {probs}")
                return False

            total = sum(probs.values())
            if not (0.98 <= total <= 1.02):
                print(f"[FAIL] Step {entry.get('step', idx)}: Probability sum {total:.6f} outside [0.98, 1.02]")
                return False

            chosen = decision.get("chosen")
            confidence = decision.get("confidence")
            if confidence != probs.get(chosen):
                print(f"[FAIL] Step {entry.get('step', idx)}: Confidence mismatch: {confidence} != {probs.get(chosen)}")
                return False

    print("[PASS] All compliance invariants (I1-I6) verified successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify official competition log file")
    parser.add_argument("log_file", type=str, help="Path to decision_log_<session_id>.jsonl")
    args = parser.parse_args()

    success = verify(args.log_file)
    sys.exit(0 if success else 1)
