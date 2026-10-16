"""Log Verification Tool — Genesis 2.0 Track EVA 02.

Strict verification CLI checking:
  [ ] every line is valid JSON
  [ ] exactly the 12 required top-level keys
  [ ] exactly the 5 decision keys
  [ ] probabilities has exactly up/down/left/right
  [ ] all probabilities finite and in [0,1]
  [ ] 0.98 <= sum <= 1.02
  [ ] chosen == action
  [ ] confidence == probabilities[chosen] EXACTLY
  [ ] source in {laya, fallback}
  [ ] result in {applied, rejected}
  [ ] step sequence over APPLIED lines == 1..N with no gaps
  [ ] every timing value is a non-negative int
  [ ] ts parses as UTC ISO8601
  [ ] hash chain re-derives correctly for every line
  [ ] prev_hash[0] == "GENESIS-<session_id>"
  [ ] fallback_count / accepted_count <= 0.10
  [ ] if --final given: final score/game/step match the last applied line

Exit code 0 on PASS, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path when invoked directly as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.official_logger import canonical_json, compute_sha256_hash


REQUIRED_TOP_LEVEL_KEYS = {
    "session_id",
    "game",
    "step",
    "state_before",
    "action",
    "state_after",
    "result",
    "decision",
    "timing",
    "ts",
    "prev_hash",
    "hash",
}

REQUIRED_DECISION_KEYS = {
    "chosen",
    "confidence",
    "probabilities",
    "source",
}

REQUIRED_DIRECTIONS = {"up", "down", "left", "right"}


def verify_log(log_path: str, final_state_path: str | None = None) -> bool:
    path = Path(log_path)
    print(f"\n{'='*70}")
    print(f"  Genesis 2.0 Log Verification Audit: {path.name}")
    print(f"{'='*70}\n")

    if not path.is_file():
        print(f"[FAIL] Log file not found: {log_path}")
        return False

    checks: list[tuple[str, bool, str]] = []
    lines: list[dict[str, Any]] = []

    # Check 1: Every line is valid JSON
    line_err = None
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                lines.append(data)
            except Exception as exc:
                line_err = f"Line {idx} invalid JSON: {exc}"
                break

    checks.append(("Every line is valid JSON", line_err is None, line_err or f"{len(lines)} lines parsed"))
    if line_err or not lines:
        _print_summary(checks)
        return False

    session_id = lines[0].get("session_id", "")
    expected_seed = f"GENESIS-{session_id}"

    # Check 2: exactly 12 top-level keys
    key_err = None
    for idx, row in enumerate(lines, start=1):
        actual_keys = set(row.keys())
        if actual_keys != REQUIRED_TOP_LEVEL_KEYS:
            diff = actual_keys.symmetric_difference(REQUIRED_TOP_LEVEL_KEYS)
            key_err = f"Line {idx}: top-level keys mismatch (diff: {diff})"
            break
    checks.append(("Exactly 12 required top-level keys", key_err is None, key_err or "All lines match"))

    # Check 3: decision keys
    dec_err = None
    for idx, row in enumerate(lines, start=1):
        dec = row.get("decision", {})
        if not isinstance(dec, dict):
            dec_err = f"Line {idx}: decision is not a dict"
            break
        # Accept decision having chosen, confidence, probabilities, source (and optional action / questions)
        if not REQUIRED_DECISION_KEYS.issubset(set(dec.keys())):
            dec_err = f"Line {idx}: missing decision keys {REQUIRED_DECISION_KEYS - set(dec.keys())}"
            break
    checks.append(("Decision object contains required keys", dec_err is None, dec_err or "All lines match"))

    # Check 4: probabilities directions
    prob_keys_err = None
    for idx, row in enumerate(lines, start=1):
        probs = row.get("decision", {}).get("probabilities", {})
        if set(probs.keys()) != REQUIRED_DIRECTIONS:
            prob_keys_err = f"Line {idx}: directions != {REQUIRED_DIRECTIONS} (got {set(probs.keys())})"
            break
    checks.append(("Probabilities has exactly up/down/left/right", prob_keys_err is None, prob_keys_err or "All lines match"))

    # Check 5: all probabilities finite and in [0, 1]
    prob_val_err = None
    for idx, row in enumerate(lines, start=1):
        probs = row.get("decision", {}).get("probabilities", {})
        for k, v in probs.items():
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or not (0.0 <= v <= 1.0):
                prob_val_err = f"Line {idx}: probability {k}={v} not in [0.0, 1.0]"
                break
        if prob_val_err:
            break
    checks.append(("All probabilities finite and in [0,1]", prob_val_err is None, prob_val_err or "All lines match"))

    # Check 6: sum in [0.98, 1.02]
    sum_err = None
    for idx, row in enumerate(lines, start=1):
        probs = row.get("decision", {}).get("probabilities", {})
        total = sum(probs.values())
        if not (0.98 <= total <= 1.02):
            sum_err = f"Line {idx}: sum={total:.6f} outside [0.98, 1.02]"
            break
    checks.append(("0.98 <= sum(probabilities) <= 1.02", sum_err is None, sum_err or "All lines match"))

    # Check 7: chosen == action
    chosen_err = None
    for idx, row in enumerate(lines, start=1):
        chosen = row.get("decision", {}).get("chosen")
        action = row.get("action")
        if chosen != action:
            chosen_err = f"Line {idx}: chosen ({chosen}) != action ({action})"
            break
    checks.append(("chosen == action", chosen_err is None, chosen_err or "All lines match"))

    # Check 8: confidence == probabilities[chosen] EXACTLY
    conf_err = None
    for idx, row in enumerate(lines, start=1):
        dec = row.get("decision", {})
        chosen = dec.get("chosen")
        conf = dec.get("confidence")
        expected_conf = dec.get("probabilities", {}).get(chosen)
        if conf != expected_conf:
            conf_err = f"Line {idx}: confidence ({conf}) != probabilities[{chosen}] ({expected_conf})"
            break
    checks.append(("confidence == probabilities[chosen] EXACTLY", conf_err is None, conf_err or "All lines match"))

    # Check 9: source in {laya, fallback}
    src_err = None
    for idx, row in enumerate(lines, start=1):
        src = row.get("decision", {}).get("source")
        if src not in ("laya", "fallback"):
            src_err = f"Line {idx}: invalid source '{src}'"
            break
    checks.append(("source in {laya, fallback}", src_err is None, src_err or "All lines match"))

    # Check 10: result in {applied, rejected}
    res_err = None
    for idx, row in enumerate(lines, start=1):
        res = row.get("result")
        if res not in ("applied", "rejected"):
            res_err = f"Line {idx}: invalid result '{res}'"
            break
    checks.append(("result in {applied, rejected}", res_err is None, res_err or "All lines match"))

    # Check 11: step sequence over APPLIED lines == 1..N with no gaps
    step_err = None
    applied_steps = [r.get("step") for r in lines if r.get("result") == "applied"]
    expected_steps = list(range(1, len(applied_steps) + 1))
    if applied_steps != expected_steps:
        step_err = f"Applied steps {applied_steps[:10]}... != {expected_steps[:10]}..."
    checks.append(("Step sequence over APPLIED lines == 1..N (gapless)", step_err is None, step_err or f"{len(applied_steps)} steps verified"))

    # Check 12: timing values non-negative int
    timing_err = None
    for idx, row in enumerate(lines, start=1):
        t = row.get("timing", {})
        for tk in ("decide_ms", "api_ms", "turn_ms"):
            tv = t.get(tk)
            if not isinstance(tv, int) or tv < 0:
                timing_err = f"Line {idx}: timing.{tk}={tv} invalid (must be non-negative int)"
                break
        if timing_err:
            break
    checks.append(("Timing values non-negative ints", timing_err is None, timing_err or "All lines match"))

    # Check 13: ts parses as UTC ISO8601
    ts_err = None
    for idx, row in enumerate(lines, start=1):
        ts_str = row.get("ts", "")
        try:
            datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts_err = f"Line {idx}: ts '{ts_str}' is not valid ISO8601"
            break
    checks.append(("ts parses as UTC ISO8601", ts_err is None, ts_err or "All lines match"))

    # Check 14 & 15: Hash chain re-derives correctly for every line and seed is GENESIS-<session_id>
    hash_err = None
    seed_err = None
    if lines[0].get("prev_hash") != expected_seed:
        seed_err = f"First line prev_hash '{lines[0].get('prev_hash')}' != '{expected_seed}'"

    expected_prev = expected_seed
    for idx, row in enumerate(lines, start=1):
        if row.get("prev_hash") != expected_prev:
            hash_err = f"Line {idx}: prev_hash mismatch: expected {expected_prev[:12]}..., got {row.get('prev_hash')[:12]}..."
            break
        # Re-derive
        row_without_hash = {k: v for k, v in row.items() if k != "hash"}
        derived = compute_sha256_hash(row_without_hash, expected_prev)
        if derived != row.get("hash"):
            hash_err = f"Line {idx}: hash re-derivation failed! Expected {derived[:12]}..., recorded {row.get('hash')[:12]}..."
            break
        expected_prev = row.get("hash")

    checks.append(("prev_hash[0] == GENESIS-<session_id>", seed_err is None, seed_err or expected_seed))
    checks.append(("Hash chain re-derives correctly for every line", hash_err is None, hash_err or f"{len(lines)} hashes re-derived"))

    # Check 16: fallback_count / accepted_count <= 0.10
    accepted_count = len(applied_steps)
    fallback_count = sum(1 for r in lines if r.get("decision", {}).get("source") == "fallback")
    fallback_ratio = (fallback_count / accepted_count) if accepted_count > 0 else 0.0
    fb_err = None
    if fallback_ratio > 0.10:
        fb_err = f"Fallback ratio {fallback_ratio:.1%} exceeds 10% budget ({fallback_count}/{accepted_count})"
    checks.append(("Fallback ratio <= 0.10", fb_err is None, fb_err or f"{fallback_ratio:.1%} ({fallback_count}/{accepted_count})"))

    # Check 17: if --final given
    if final_state_path:
        final_p = Path(final_state_path)
        final_err = None
        if not final_p.is_file():
            final_err = f"Final state file not found: {final_state_path}"
        else:
            with open(final_p, "r", encoding="utf-8") as ff:
                final_data = json.load(ff)
            last_applied = [r for r in lines if r.get("result") == "applied"]
            if last_applied:
                last_line = last_applied[-1]
                # Compare step
                if final_data.get("step") is not None and final_data.get("step") != last_line.get("step"):
                    final_err = f"Final state step ({final_data.get('step')}) != last applied step ({last_line.get('step')})"
        checks.append(("Final state matches last applied line", final_err is None, final_err or "Final state verified"))

    all_passed = _print_summary(checks)
    return all_passed


def _print_summary(checks: list[tuple[str, bool, str]]) -> bool:
    all_ok = True
    first_failure = None

    for name, ok, detail in checks:
        status_str = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_ok = False
            if first_failure is None:
                first_failure = f"{name}: {detail}"
        print(f"  {status_str}  {name:<55} | {detail}")

    print("\n" + "-" * 70)
    if all_ok:
        print("[AUDIT PASSED] Log file strictly satisfies all Genesis 2.0 invariants.")
    else:
        print(f"[AUDIT FAILED] First Failure:\n  >> {first_failure}")
    print("-" * 70 + "\n")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Genesis 2.0 log verification")
    parser.add_argument("log_file", help="Path to decision_log_<session_id>.jsonl")
    parser.add_argument("--final", default=None, help="Optional path to final_state_<session_id>.json")
    args = parser.parse_args()

    ok = verify_log(args.log_file, args.final)
    sys.exit(0 if ok else 1)
