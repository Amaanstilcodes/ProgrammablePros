"""API Probe Tool — Genesis 2.0 Track EVA 02.

Safely tests network connectivity and authentication against TopGameAI Arena
endpoints without running a full game session.
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

from core.config import AppConfig


def probe_endpoint(base_url: str, api_key: str = "") -> bool:
    print(f"[*] Probing TopGameAI Arena API at: {base_url}")
    headers = {"User-Agent": "ProgrammablePros-Probe/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        t0 = time.perf_counter()
        resp = requests.get(base_url, headers=headers, timeout=5.0)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[OK] Response received in {ms:.1f}ms — HTTP Status: {resp.status_code}")
        return True
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] Connection failed: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe TopGameAI API connectivity")
    parser.add_argument("--url", default="", help="Override API base URL")
    parser.add_argument("--api-key", default="", help="Override API Key")
    args = parser.parse_args()

    config = AppConfig.load()
    target_url = args.url or config.topgame_api_url
    target_key = args.api_key or config.topgame_api_key

    ok = probe_endpoint(target_url, target_key)
    sys.exit(0 if ok else 1)
