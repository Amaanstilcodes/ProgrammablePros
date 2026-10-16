"""Export Run Tool — Genesis 2.0 Track EVA 02.

Packages and validates a completed competition run into submission/verified/<session_id>/
generating checksums for judges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from core.official_logger import OfficialLogger


def export_session(session_path: str, destination: str = "submission/verified") -> bool:
    src = Path(session_path)
    if not src.is_dir():
        print(f"[FAIL] Session directory does not exist: {src}")
        return False

    log_files = list(src.glob("decision_log_*.jsonl"))
    final_files = list(src.glob("final_state_*.json"))
    manifest_files = list(src.glob("manifest.json"))

    if not log_files or not final_files:
        print("[FAIL] Missing decision_log or final_state in session directory.")
        return False

    # Verify log integrity first
    is_valid, msg = OfficialLogger.verify_log_file(log_files[0])
    if not is_valid:
        print(f"[FAIL] Cannot export unverified run: {msg}")
        return False

    dest = Path(destination) / src.name
    dest.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)

    # Compute checksum of official log
    sha = hashlib.sha256(log_files[0].read_bytes()).hexdigest()
    checksum_file = dest / "CHECKSUM.sha256"
    checksum_file.write_text(f"{sha}  {log_files[0].name}\n", encoding="utf-8")

    print(f"[SUCCESS] Exported verified run to: {dest}")
    print(f"          Log SHA-256: {sha}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a verified competition session")
    parser.add_argument("session_dir", type=str, help="Path to session folder")
    parser.add_argument("--dest", type=str, default="submission/verified")
    args = parser.parse_args()

    export_session(args.session_dir, args.dest)
