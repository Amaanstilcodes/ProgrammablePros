"""API Discovery & Probe Tool — Genesis 2.0 Track EVA 02.

A manual, human-driven discovery tool for TopGameAI Arena endpoints.
Never auto-plays moves. One invocation = one human-requested HTTP call.
Saves all captured responses into docs/fixtures/ for contract verification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests



def _get_env_config() -> dict[str, str]:
    """Load discovery configuration from environment."""
    base_url = (
        os.getenv("TOPGAMEAI_BASE_URL")
        or os.getenv("TOPGAME_API_URL")
        or "https://topgameai.com"
    ).rstrip("/")
    api_key = os.getenv("TOPGAMEAI_API_KEY") or os.getenv("TOPGAME_API_KEY") or ""
    agent_name = os.getenv("AGENT_NAME") or "ProgrammablePros"
    return {
        "base_url": base_url,
        "api_key": api_key,
        "agent_name": agent_name,
    }


def _redacted_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of headers with authorization redacted."""
    redacted = dict(headers)
    for k in list(redacted.keys()):
        if k.lower() in ("authorization", "x-api-key", "api-key"):
            redacted[k] = "[REDACTED]"
    return redacted


def _inspect_payload_keys(data: Any) -> list[str]:
    """Check presence of standard arena keys."""
    target_keys = [
        "session_id",
        "state",
        "state_before",
        "state_after",
        "score",
        "game_over",
        "rejected",
        "step",
        "direction",
        "board",
        "food",
    ]
    found = []
    if isinstance(data, dict):
        for k in target_keys:
            if k in data or (isinstance(data.get("state"), dict) and k in data["state"]):
                found.append(k)
    return found


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_discover(base_url: str, api_key: str, game: str) -> None:
    """GET discovery endpoints without making any assumptions."""
    out_dir = _ensure_dir(Path("docs/fixtures/discovery"))
    paths = [
        ("/", "root"),
        ("/docs", "docs"),
        ("/openapi.json", "openapi"),
        ("/api/docs", "api_docs"),
        (f"/arena/{game}", f"arena_{game}"),
        (f"/api/arena/{game}", f"api_arena_{game}"),
    ]

    headers = {"User-Agent": "ProgrammablePros-Probe/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"[*] Discovering endpoints for game '{game}' on base URL: {base_url}")
    session = requests.Session()

    for path, name in paths:
        target = urljoin(base_url + "/", path.lstrip("/"))
        print(f"\n--- GET {target} ---")
        t0 = time.perf_counter()
        try:
            resp = session.get(target, headers=headers, timeout=10.0)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            content_type = resp.headers.get("Content-Type", "unknown")
            print(f"HTTP Status: {resp.status_code}")
            print(f"Content-Type: {content_type}")
            print(f"Latency: {latency_ms:.1f}ms")

            snippet = resp.text[:2000]
            print(f"Preview (first 2000 chars):\n{snippet}\n")

            out_file = out_dir / f"{name}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(
                    f"URL: {target}\n"
                    f"HTTP Status: {resp.status_code}\n"
                    f"Content-Type: {content_type}\n"
                    f"Latency_ms: {latency_ms:.2f}\n"
                    f"Headers: {json.dumps(dict(resp.headers), indent=2)}\n\n"
                    f"--- BODY ---\n"
                    f"{resp.text}\n"
                )
            print(f"-> Saved discovery response to {out_file}")

        except requests.exceptions.RequestException as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[FAIL] Request failed ({latency_ms:.1f}ms): {exc}")


def run_start(
    base_url: str,
    api_key: str,
    game: str,
    endpoint: str,
    payload_template: str,
) -> None:
    """Send explicit start POST and save full request and response fixtures."""
    if not endpoint:
        print("[ERROR] --endpoint is required for 'start' (e.g. --endpoint /api/arena/snake/start)")
        sys.exit(1)
    if not payload_template:
        print("[ERROR] --payload-template is required for 'start' (e.g. --payload-template '{\"agent\":\"ProgrammablePros\"}')")
        sys.exit(1)

    try:
        payload = json.loads(payload_template)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in --payload-template: {exc}")
        sys.exit(1)

    target_url = urljoin(base_url + "/", endpoint.lstrip("/"))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ProgrammablePros-Probe/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"[*] Sending START request to: {target_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    t0 = time.perf_counter()
    resp = requests.post(target_url, json=payload, headers=headers, timeout=10.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nHTTP Status: {resp.status_code} ({latency_ms:.1f}ms)")
    try:
        resp_json = resp.json()
        print(f"Response Body:\n{json.dumps(resp_json, indent=2)}")
    except Exception:
        resp_json = None
        print(f"Response Text:\n{resp.text}")

    detected_keys = _inspect_payload_keys(resp_json)
    print(f"Detected relevant keys: {detected_keys}")

    out_dir = _ensure_dir(Path(f"docs/fixtures/{game}"))
    resp_file = out_dir / "start.json"
    req_file = out_dir / "start.request.json"

    with open(resp_file, "w", encoding="utf-8") as f:
        if resp_json is not None:
            json.dump(resp_json, f, indent=2)
        else:
            f.write(resp.text)

    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "POST",
                "url": target_url,
                "headers": _redacted_headers(headers),
                "payload": payload,
                "http_status": resp.status_code,
                "latency_ms": round(latency_ms, 2),
                "detected_keys": detected_keys,
            },
            f,
            indent=2,
        )

    print(f"-> Saved response fixture: {resp_file}")
    print(f"-> Saved request fixture:  {req_file}")


def run_move(
    base_url: str,
    api_key: str,
    game: str,
    session_id: str,
    move: str,
    endpoint: str,
) -> None:
    """Send a single move and save the fixture."""
    if not session_id:
        print("[ERROR] --session is required for 'move'")
        sys.exit(1)
    if not move:
        print("[ERROR] --move is required for 'move' (e.g. --move up)")
        sys.exit(1)

    ep = endpoint or f"/api/arena/{game}/move"
    target_url = urljoin(base_url + "/", ep.lstrip("/"))

    payload = {"session_id": session_id, "action": move}
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ProgrammablePros-Probe/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"[*] Sending MOVE '{move}' to: {target_url} (Session: {session_id})")

    t0 = time.perf_counter()
    resp = requests.post(target_url, json=payload, headers=headers, timeout=10.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nHTTP Status: {resp.status_code} ({latency_ms:.1f}ms)")
    try:
        resp_json = resp.json()
        print(f"Response:\n{json.dumps(resp_json, indent=2)}")
    except Exception:
        resp_json = None
        print(f"Response Text:\n{resp.text}")

    out_dir = _ensure_dir(Path(f"docs/fixtures/{game}"))

    # Auto-index move_n.json
    existing = list(out_dir.glob("move_*.json"))
    idx = len(existing) + 1
    move_file = out_dir / f"move_{idx}.json"

    with open(move_file, "w", encoding="utf-8") as f:
        if resp_json is not None:
            json.dump(resp_json, f, indent=2)
        else:
            f.write(resp.text)

    is_rejected = False
    if isinstance(resp_json, dict):
        if resp_json.get("result") == "rejected" or resp_json.get("rejected") is True:
            is_rejected = True

    if is_rejected:
        rejected_file = out_dir / "move_rejected.json"
        with open(rejected_file, "w", encoding="utf-8") as f:
            json.dump(resp_json, f, indent=2)
        print(f"-> Captured REJECTED move fixture: {rejected_file}")

    print(f"-> Saved move fixture: {move_file}")


def run_state(
    base_url: str,
    api_key: str,
    game: str,
    session_id: str,
    endpoint: str,
) -> None:
    """GET current state and save fixture."""
    if not session_id:
        print("[ERROR] --session is required for 'state'")
        sys.exit(1)

    ep = endpoint or f"/api/arena/{game}/state"
    target_url = urljoin(base_url + "/", ep.lstrip("/"))
    headers = {"User-Agent": "ProgrammablePros-Probe/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"[*] Getting STATE for session {session_id} from: {target_url}")

    t0 = time.perf_counter()
    resp = requests.get(target_url, params={"session_id": session_id}, headers=headers, timeout=10.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nHTTP Status: {resp.status_code} ({latency_ms:.1f}ms)")
    try:
        resp_json = resp.json()
        print(f"Response:\n{json.dumps(resp_json, indent=2)}")
    except Exception:
        resp_json = None
        print(f"Response Text:\n{resp.text}")

    out_dir = _ensure_dir(Path(f"docs/fixtures/{game}"))
    state_file = out_dir / "state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        if resp_json is not None:
            json.dump(resp_json, f, indent=2)
        else:
            f.write(resp.text)

    print(f"-> Saved state fixture: {state_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TopGameAI Arena manual probe tool")
    parser.add_argument("--game", choices=["snake", "2048"], default="snake", help="Target arena game")
    parser.add_argument(
        "--action",
        choices=["discover", "start", "move", "state"],
        required=True,
        help="Action to execute",
    )
    parser.add_argument("--session", default="", help="Session ID for move/state")
    parser.add_argument("--move", default="", help="Move direction (up/down/left/right)")
    parser.add_argument("--endpoint", default="", help="Explicit endpoint path")
    parser.add_argument("--payload-template", default="", help="JSON string payload for start")
    parser.add_argument("--base-url", default="", help="Override base URL")
    parser.add_argument("--api-key", default="", help="Override API Key")

    args = parser.parse_args()
    env = _get_env_config()
    base_url = args.base_url or env["base_url"]
    api_key = args.api_key or env["api_key"]

    if args.action == "discover":
        run_discover(base_url, api_key, args.game)
    elif args.action == "start":
        run_start(base_url, api_key, args.game, args.endpoint, args.payload_template)
    elif args.action == "move":
        run_move(base_url, api_key, args.game, args.session, args.move, args.endpoint)
    elif args.action == "state":
        run_state(base_url, api_key, args.game, args.session, args.endpoint)


if __name__ == "__main__":
    main()
