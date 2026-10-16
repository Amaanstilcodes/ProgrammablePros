"""Live Compliance Viewer — Genesis 2.0 Track EVA 02.

Streamlit live telemetry and audit dashboard:
- Dual-log audit: Official Hash-Chained Log & Rich Internal Trace
- Real-time SHA-256 cryptographic verification
- Board visualizer: Snake ASCII arena & 2048 dynamic tile grid
- Compliance metrics: Source badges, timing breakdown, probability distribution
- Chronological decision audit table
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.official_logger import OfficialLogger


def setup_ui() -> None:
    st.set_page_config(
        page_title="Genesis 2.0 Compliance Viewer | Track EVA 02",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .badge-laya {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
        }
        .badge-fallback {
            background: rgba(245, 158, 11, 0.2);
            border: 1px solid #f59e0b;
            color: #fbbf24;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
        }
        .tile-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            background: #1e293b;
            padding: 12px;
            border-radius: 8px;
            max-width: 320px;
        }
        .tile-cell {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 700;
            border-radius: 6px;
            background: #0f172a;
            color: #94a3b8;
        }
        .tile-2 { background: #334155; color: #fff; }
        .tile-4 { background: #475569; color: #fff; }
        .tile-8 { background: #d97706; color: #fff; }
        .tile-16 { background: #ea580c; color: #fff; }
        .tile-32 { background: #dc2626; color: #fff; }
        .tile-64 { background: #b91c1c; color: #fff; }
        .tile-128 { background: #eab308; color: #000; }
        .tile-256 { background: #facc15; color: #000; }
        .tile-512 { background: #38bdf8; color: #000; }
        .tile-1024 { background: #818cf8; color: #fff; }
        .tile-2048 { background: #a855f7; color: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1)
def load_lines(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))
    except Exception:
        pass
    return records


def render_snake_ascii(state: dict[str, Any], width: int = 16, height: int = 16) -> str:
    grid = [["· " for _ in range(width)] for _ in range(height)]
    food = state.get("food", [5, 5])
    if isinstance(food, (list, tuple)) and len(food) >= 2:
        fx = int(food[0][0]) if isinstance(food[0], (list, tuple)) else int(food[0])
        fy = int(food[0][1]) if isinstance(food[0], (list, tuple)) else int(food[1])
        if 0 <= fx < width and 0 <= fy < height:
            grid[fy][fx] = "★ "

    snake = state.get("snake", [])
    if isinstance(snake, (list, tuple)):
        for idx, pt in enumerate(snake):
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < width and 0 <= y < height:
                    grid[y][x] = "◆ " if idx == 0 else "■ "

    rows = ["".join(r) for r in grid]
    return "\n".join(rows)


def render_2048_html(board: list[list[int]]) -> str:
    cells = []
    for row in board:
        for val in row:
            tile_cls = f"tile-{val}" if val in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) else ""
            txt = str(val) if val > 0 else ""
            cells.append(f'<div class="tile-cell {tile_cls}">{txt}</div>')
    return f'<div class="tile-grid">{"".join(cells)}</div>'


def main() -> None:
    setup_ui()

    st.sidebar.title("Telemetry Controls")
    # Discover available session logs
    discovered_logs = list(Path("logs").glob("**/*.jsonl"))
    log_options = [str(p) for p in discovered_logs] if discovered_logs else ["decision_log.jsonl"]
    default_log = log_options[0] if log_options else "decision_log.jsonl"

    selected_log = st.sidebar.selectbox("Select Log Stream", options=log_options, index=0)
    auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh (sec)", 1.0, 5.0, 2.0, 0.5)

    if st.sidebar.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.title("🛡️ TopGameAI Compliance Telemetry Viewer")
    st.caption("Live Invariant Audit · Track EVA 02 (Snake & 2048)")

    records = load_lines(selected_log)
    if not records:
        st.info(f"Waiting for records in `{selected_log}`...")
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
        return

    latest = records[-1]
    game = latest.get("game", "snake")
    step = latest.get("step", 0)
    action = str(latest.get("action", "")).upper()
    result = str(latest.get("result", "")).upper()
    decision = latest.get("decision", {})
    source = decision.get("source", "laya")
    conf = decision.get("confidence", 0.0)
    timing = latest.get("timing", {})
    state_after = latest.get("state_after", {})
    probs = decision.get("probabilities", {})

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Game / Step", f"{game.upper()} · #{step}")
    m2.metric("Action (Result)", f"{action} ({result})")
    with m3:
        st.caption("Decision Source")
        if source == "laya":
            st.markdown('<span class="badge-laya">🟢 LAYA</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-fallback">🟠 FALLBACK</span>', unsafe_allow_html=True)
    m4.metric("Confidence", f"{conf:.4f}")

    # Hash verification
    is_valid, msg = OfficialLogger.verify_log_file(selected_log)
    if is_valid:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")

    # Board + Charts
    c_left, c_right = st.columns([1, 1.2])
    with c_left:
        st.subheader("Arena View")
        if game == "snake":
            st.code(render_snake_ascii(state_after), language="text")
        else:
            board = state_after.get("board", [[0] * 4 for _ in range(4)])
            st.markdown(render_2048_html(board), unsafe_allow_html=True)

    with c_right:
        st.subheader("Probability Simplex")
        if probs:
            df = pd.DataFrame([{"Action": k.upper(), "Probability": v} for k, v in probs.items()]).set_index("Action")
            st.bar_chart(df)
            prob_sum = sum(probs.values())
            st.caption(f"**Sum:** `{prob_sum:.6f}` (Mandated: [0.98, 1.02])")

        t1, t2, t3 = st.columns(3)
        t1.metric("Decide", f"{timing.get('decide_ms', 0)} ms")
        t2.metric("API", f"{timing.get('api_ms', 0)} ms")
        t3.metric("Turn", f"{timing.get('turn_ms', 0)} ms")

    st.divider()
    st.subheader("Decision Audit Trail (Last 20)")
    feed = []
    for r in records[-20:]:
        dec = r.get("decision", {})
        feed.append({
            "Step": r.get("step"),
            "Action": str(r.get("action")).upper(),
            "Result": str(r.get("result")).upper(),
            "Confidence": f"{dec.get('confidence', 0.0):.4f}",
            "Source": dec.get("source", "").upper(),
            "Turn (ms)": r.get("timing", {}).get("turn_ms", 0),
            "Hash": r.get("hash", "")[:12] + "…",
        })
    if feed:
        st.dataframe(pd.DataFrame(feed), use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
