"""Streamlit Live Decision & Compliance Viewer — Genesis 2.0 Track EVA 02.

Real-time telemetry and compliance audit dashboard for TopGameAI Arena (Snake & 2048):
- Auto-refreshing live visualization connected to decision_log.jsonl
- Real-time cryptographic SHA-256 hash chain verification
- Dynamic game rendering: ASCII grid for Snake & styled tile matrix for 2048
- Compliance telemetry: Laya vs Fallback source badges, timing metrics, and probability distribution
- Chronological feed of the last 20 steps
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from official_logger import OfficialLogger

# ─────────────────────────────────────────────────────────────────────────────
# UI Configuration & Styling
# ─────────────────────────────────────────────────────────────────────────────

def setup_page() -> None:
    """Configure Streamlit layout, dark theme accents, and custom CSS."""
    st.set_page_config(
        page_title="Genesis 2.0 Live Viewer | Track EVA 02",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Base typography & palette */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Top header container */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border: 1px solid #312e81;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .main-title {
            font-size: 26px;
            font-weight: 700;
            color: #f8fafc;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .main-sub {
            font-size: 14px;
            color: #94a3b8;
            margin-top: 6px;
            margin-bottom: 0;
        }

        /* Metrics cards */
        .metric-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #f3f4f6;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Source badges */
        .badge-laya {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 6px 14px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 14px;
            display: inline-block;
            letter-spacing: 0.05em;
        }
        .badge-fallback {
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid #f59e0b;
            color: #fbbf24;
            padding: 6px 14px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 14px;
            display: inline-block;
            letter-spacing: 0.05em;
        }

        /* Hash chain banner */
        .hash-valid {
            background: rgba(6, 78, 59, 0.4);
            border: 1px solid #059669;
            color: #6ee7b7;
            border-radius: 8px;
            padding: 10px 14px;
            font-weight: 600;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .hash-broken {
            background: rgba(127, 29, 29, 0.4);
            border: 1px solid #dc2626;
            color: #fca5a5;
            border-radius: 8px;
            padding: 10px 14px;
            font-weight: 600;
            font-size: 13px;
        }

        /* 2048 Tile styling */
        .tile-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            background: #1e293b;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #334155;
            max-width: 360px;
            margin: auto;
        }
        .tile-cell {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 700;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            background: #0f172a;
            color: #64748b;
        }
        .tile-2 { background: #334155; color: #f8fafc; }
        .tile-4 { background: #475569; color: #f8fafc; }
        .tile-8 { background: #d97706; color: #fff; }
        .tile-16 { background: #ea580c; color: #fff; }
        .tile-32 { background: #dc2626; color: #fff; }
        .tile-64 { background: #b91c1c; color: #fff; }
        .tile-128 { background: #eab308; color: #000; box-shadow: 0 0 10px #eab308; }
        .tile-256 { background: #facc15; color: #000; box-shadow: 0 0 12px #facc15; }
        .tile-512 { background: #38bdf8; color: #000; box-shadow: 0 0 14px #38bdf8; }
        .tile-1024 { background: #818cf8; color: #fff; box-shadow: 0 0 16px #818cf8; }
        .tile-2048 { background: #a855f7; color: #fff; box-shadow: 0 0 20px #a855f7; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe Cached Data Reader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1)
def load_log_records(log_file_path: str) -> list[dict[str, Any]]:
    """Safely read JSONL decision records with TTL cache to prevent lock conflicts.

    Args:
        log_file_path: File system path to the .jsonl log.

    Returns:
        List of parsed JSON record dictionaries.
    """
    path = Path(log_file_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))
    except Exception:
        # File might be partially written by compliant_runner mid-turn
        pass

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Board Visualizers
# ─────────────────────────────────────────────────────────────────────────────

def render_snake_ascii(state: dict[str, Any], width: int = 16, height: int = 16) -> str:
    """Generate a clean ASCII grid representation of the Snake board."""
    grid = [["· " for _ in range(width)] for _ in range(height)]

    # Food position
    food_data = state.get("food", [5, 5])
    if isinstance(food_data, (list, tuple)) and len(food_data) >= 2:
        if isinstance(food_data[0], (list, tuple)):
            fx, fy = int(food_data[0][0]), int(food_data[0][1])
        else:
            fx, fy = int(food_data[0]), int(food_data[1])
        if 0 <= fx < width and 0 <= fy < height:
            grid[fy][fx] = "★ "

    # Snake body
    snake_body = state.get("snake", [])
    if isinstance(snake_body, (list, tuple)):
        for idx, pt in enumerate(snake_body):
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < width and 0 <= y < height:
                    grid[y][x] = "◆ " if idx == 0 else "■ "

    border_top = "┌" + "──" * width + "┐"
    border_bot = "└" + "──" * width + "┘"
    rows = [f"│{''.join(row)}│" for row in grid]
    return "\n".join([border_top] + rows + [border_bot])


def render_2048_html(board: list[list[int]]) -> str:
    """Render an HTML grid of the 2048 tiles."""
    html_cells = []
    for row in board:
        for val in row:
            tile_class = f"tile-{val}" if val in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) else ""
            display_text = str(val) if val > 0 else ""
            html_cells.append(f'<div class="tile-cell {tile_class}">{display_text}</div>')

    return f'<div class="tile-grid">{"".join(html_cells)}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main Streamlit dashboard lifecycle."""
    setup_page()

    # Sidebar Controls
    st.sidebar.title("⚙️ Telemetry Controls")
    log_file = st.sidebar.text_input("Log File", value="decision_log.jsonl")
    auto_refresh = st.sidebar.checkbox("Live Auto-Refresh", value=True)
    refresh_rate = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        ### 🛡️ Genesis 2.0 Invariants
        - **Track:** EVA 02 (TopGameAI Arena)
        - **Stages:** Stage 1 (Snake) & Stage 2 (2048)
        - **Log Format:** Strict SHA-256 JSONL
        - **Probabilities:** 4 keys, Sum ∈ [0.98, 1.02]
        - **Fallback Cap:** ≤ 10%
        """
    )

    if st.sidebar.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    # Header
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">
                <span>🤖 TopGameAI Autonomous Live Dashboard</span>
            </div>
            <div class="main-sub">
                Genesis 2.0 Track EVA 02 — Real-time telemetry, calibration, and cryptographic audit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Read records
    records = load_log_records(log_file)

    if not records:
        st.info(
            f"⏳ **Waiting for logs in `{log_file}`...**\n\n"
            "Run `python compliant_runner.py` in your terminal to begin generating verified moves."
        )
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
        return

    latest = records[-1]
    game_type = str(latest.get("game", "snake")).lower()
    step_num = latest.get("step", len(records))
    action = str(latest.get("action", "NONE")).upper()
    result = str(latest.get("result", "applied")).upper()
    decision = latest.get("decision", {})
    source = str(decision.get("source", "laya")).lower()
    confidence = float(decision.get("confidence", 0.0))
    probs = decision.get("probabilities", {})
    timing = latest.get("timing", {})
    state_after = latest.get("state_after", {})
    questions = decision.get("questions", [])

    # Top Telemetry Row
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Game & Step</div>
                <div class="metric-value">{game_type.upper()} · #{step_num}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Action & Outcome</div>
                <div class="metric-value">{action} <span style="font-size:16px;color:#10b981;">({result})</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t3:
        badge_html = (
            '<span class="badge-laya">🟢 LAYA INFERENCE</span>'
            if source == "laya"
            else '<span class="badge-fallback">🟠 SAFETY FALLBACK</span>'
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Decision Source</div>
                <div style="margin-top:6px;">{badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Chosen Confidence</div>
                <div class="metric-value">{confidence:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Hash Chain Status & Audit Bar
    is_valid, verify_msg = OfficialLogger.verify_log_file(log_file)
    if is_valid:
        st.markdown(
            f"""
            <div class="hash-valid">
                <span>🛡️</span> <b>Hash Chain Verified:</b> {verify_msg}
                <span style="margin-left:auto; font-family:'JetBrains Mono'; font-size:11px; opacity:0.8;">
                    Latest Hash: {latest.get('hash', '')[:20]}…
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="hash-broken">
                <span>⚠️</span> <b>Hash Chain Invalidation Warning:</b> {verify_msg}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Main Visualization Columns
    col_left, col_right = st.columns([1.1, 1.2])

    with col_left:
        st.subheader("🎮 Live Arena State")

        if game_type == "snake":
            snake_body = state_after.get("snake", [])
            snake_len = len(snake_body) if isinstance(snake_body, list) else 1
            snake_score = state_after.get("score", 0)
            snake_dir = str(state_after.get("direction", "N/A")).upper()

            sub_c1, sub_c2, sub_c3 = st.columns(3)
            sub_c1.metric("Score", snake_score)
            sub_c2.metric("Length", snake_len)
            sub_c3.metric("Direction", snake_dir)

            ascii_grid = render_snake_ascii(state_after)
            st.caption("ASCII Arena Board (◆ Head, ■ Body, ★ Food):")
            st.code(ascii_grid, language="text")

        elif game_type == "2048":
            board_matrix = state_after.get("board") or [[0] * 4 for _ in range(4)]
            score_2048 = state_after.get("score", 0)
            max_tile = state_after.get("max_tile", max(max(r) for r in board_matrix))

            sub_c1, sub_c2 = st.columns(2)
            sub_c1.metric("Current Score", score_2048)
            sub_c2.metric("Max Tile", max_tile)

            st.caption("Dynamic Board Grid:")
            st.markdown(render_2048_html(board_matrix), unsafe_allow_html=True)

    with col_right:
        st.subheader("📊 Calibrated Probabilities & Latency")

        # Probability distribution bar chart
        if probs:
            prob_df = pd.DataFrame(
                [{"Action": k.upper(), "Probability": v} for k, v in probs.items()]
            ).set_index("Action")
            st.bar_chart(prob_df, height=220)

            # Sum verification indicator
            prob_sum = sum(probs.values())
            sum_color = "green" if 0.98 <= prob_sum <= 1.02 else "red"
            st.caption(
                f"**Probability Sum:** <span style='color:{sum_color}; font-weight:bold;'>"
                f"{prob_sum:.6f}</span> (Mandated: 0.98 to 1.02)",
                unsafe_allow_html=True,
            )

        # 3-Column Timing Metrics
        st.markdown("##### ⏱️ Latency Telemetry (per PDF §6)")
        time_c1, time_c2, time_c3 = st.columns(3)
        with time_c1:
            st.metric("Decide Time", f"{timing.get('decide_ms', 0):.0f} ms")
        with time_c2:
            st.metric("API RTT", f"{timing.get('api_ms', 0):.0f} ms")
        with time_c3:
            st.metric("Total Turn", f"{timing.get('turn_ms', 0):.0f} ms")

        # Active Hypothesis Questions
        if questions:
            with st.expander(f"🔎 Formulated Hypothesis Questions ({len(questions)} queries)", expanded=False):
                for q_idx, q_text in enumerate(questions, start=1):
                    st.markdown(f"**Q{q_idx}:** `{q_text}`")

    st.divider()

    # Chronological Decision Feed (Last 20 Steps)
    st.subheader("📜 Chronological Decision Audit Feed")
    feed_rows = []
    recent_records = records[-20:]

    for r in recent_records:
        r_step = r.get("step")
        r_action = str(r.get("action", "")).upper()
        r_result = str(r.get("result", "")).upper()
        r_dec = r.get("decision", {})
        r_conf = f"{r_dec.get('confidence', 0.0):.4f}"
        r_src = r_dec.get("source", "laya").upper()
        r_timing = f"{r.get('timing', {}).get('turn_ms', 0):.0f} ms"
        r_hash = r.get("hash", "")[:12] + "…"

        feed_rows.append({
            "Step": r_step,
            "Action": r_action,
            "Result": r_result,
            "Confidence": r_conf,
            "Source": r_src,
            "Turn Time": r_timing,
            "Hash Prefix": r_hash,
        })

    if feed_rows:
        feed_df = pd.DataFrame(feed_rows)
        st.dataframe(feed_df, use_container_width=True, hide_index=True)

    # Auto-refresh loop
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
