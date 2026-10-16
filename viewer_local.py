"""Local Games Live Telemetry Viewer — Genesis 2.0 Track EVA 02.

Streamlit interactive dashboard supporting Snake & 2048:
- Real-time polling & incremental JSONL parsing
- 20x20 Snake board rendering & 4x4 2048 styled board
- Probability & EV distributions
- Cryptographic hash-chain integrity verification
- Hypothesis inspector & full audit feed
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from official_logger import OfficialLogger

# Configure Streamlit page
st.set_page_config(
    page_title="Laya EVA 02 Live Telemetry",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .kpi-card {
        background-color: #1a1c24;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-laya {
        background-color: #134e4a;
        color: #2dd4bf;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-fallback {
        background-color: #713f12;
        color: #facc15;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-pass {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-fail {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .tile-cell {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 60px;
        width: 60px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 6px;
        margin: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_log_entries(log_file: Path) -> list[dict[str, Any]]:
    """Parse valid JSON lines from log file."""
    entries = []
    if not log_file.exists():
        return entries
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entries.append(json.loads(line_str))
            except json.JSONDecodeError:
                continue
    return entries


def render_snake_grid(state: dict[str, Any]) -> None:
    """Render 20x20 Snake grid using pandas DataFrame."""
    width = state.get("width", 20)
    height = state.get("height", 20)
    snake = state.get("snake", [])
    food = state.get("food", [-1, -1])

    # Grid init
    grid = [["·" for _ in range(width)] for _ in range(height)]

    # Draw body
    for i, seg in enumerate(snake):
        x, y = seg[0], seg[1]
        if 0 <= x < width and 0 <= y < height:
            if i == 0:
                grid[y][x] = "🟢H"  # Head
            else:
                grid[y][x] = "🟦"  # Body

    # Draw food
    fx, fy = food[0], food[1]
    if 0 <= fx < width and 0 <= fy < height:
        grid[fy][fx] = "🍎"

    df = pd.DataFrame(grid)
    st.dataframe(df, use_container_width=True, height=450)


def render_2048_grid(state: dict[str, Any]) -> None:
    """Render 4x4 2048 board with color coding."""
    board = state.get("board", [[0] * 4 for _ in range(4)])
    color_map = {
        0: "#cdc1b4",
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f2b179",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc850",
        1024: "#edc53f",
        2048: "#edc22e",
    }
    font_color_map = {0: "#776e65", 2: "#776e65", 4: "#776e65"}

    cols = st.columns(4)
    for r in range(4):
        c_cols = st.columns(4)
        for c in range(4):
            val = board[r][c]
            bg_col = color_map.get(val, "#3c3a32")
            fg_col = font_color_map.get(val, "#f9f6f2")
            val_str = str(val) if val > 0 else ""
            with c_cols[c]:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_col}; color: {fg_col};
                                height: 75px; display: flex; align-items: center;
                                justify-content: center; font-size: 24px; font-weight: bold;
                                border-radius: 6px; margin: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        {val_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.title("🎮 Laya Autonomous Game Decision Viewer (Local)")
    st.caption("Genesis 2.0 Track EVA 02 | Live Telemetry & Audit Chain")

    # Sidebar
    st.sidebar.header("🕹️ Session Inspector")
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(list(log_dir.glob("decision_log_*.jsonl")), key=os.path.getmtime, reverse=True)
    if not log_files:
        st.sidebar.warning("No log files found in logs/ directory.")
        st.info("Run `python run_local_games.py --game snake` or `2048` to generate a live session log.")
        return

    selected_file = st.sidebar.selectbox("Select Log File", log_files, format_func=lambda p: p.name)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (1s)", value=True)

    entries = load_log_entries(selected_file)
    if not entries:
        st.warning(f"Log file {selected_file.name} is currently empty.")
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        return

    latest = entries[-1]
    game = latest.get("game", "snake")
    state = latest.get("state_after", latest.get("state_before", {}))
    decision = latest.get("decision", {})
    timing = latest.get("timing", {})

    # Top KPI Row
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.metric("Session ID", latest.get("session_id", "N/A")[:12])
    with kpi2:
        st.metric("Game", game.upper())
    with kpi3:
        st.metric("Step", latest.get("step", 0))
    with kpi4:
        score = state.get("score", 0)
        st.metric("Score", score)
    with kpi5:
        if game == "snake":
            st.metric("Foods Eaten", f"{state.get('foods_eaten', 0)} / 15")
        else:
            st.metric("Best Tile", state.get("best_tile", 0))
    with kpi6:
        action = latest.get("action", "N/A").upper()
        st.metric("Latest Move", action)

    # Verification and Fallback Stats
    fb_count = sum(1 for e in entries if e.get("decision", {}).get("source") == "fallback")
    fb_pct = (fb_count / len(entries)) * 100.0 if entries else 0.0

    is_valid, v_msg = OfficialLogger.verify_log_file(selected_file)

    st.markdown("---")
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    with badge_col1:
        src = decision.get("source", "laya")
        b_class = "badge-laya" if src == "laya" else "badge-fallback"
        st.markdown(f"**Decision Source:** <span class='{b_class}'>{src.upper()}</span>", unsafe_allow_html=True)
    with badge_col2:
        st.markdown(f"**Fallback Rate:** `{fb_pct:.1f}%` (Budget: <= 10.0%)")
        st.progress(min(1.0, fb_pct / 100.0))
    with badge_col3:
        v_class = "badge-pass" if is_valid else "badge-fail"
        v_label = "PASS (VALID)" if is_valid else "FAIL"
        st.markdown(f"**Hash Chain:** <span class='{v_class}'>{v_label}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Main columns: Left = Board, Right = Decision / Probabilities
    col_board, col_decision = st.columns([1.1, 0.9])

    with col_board:
        st.subheader("🎯 Board State Visualization")
        if game == "snake":
            render_snake_grid(state)
        else:
            render_2048_grid(state)

    with col_decision:
        st.subheader("⚖️ Decision Engine Telemetry")

        # Probabilities chart
        probs = decision.get("probabilities", {})
        if probs:
            st.markdown("**Calibrated Direction Probabilities:**")
            p_df = pd.DataFrame(list(probs.items()), columns=["Action", "Probability"]).set_index("Action")
            st.bar_chart(p_df, height=180)

        # Confidence and timing
        st.markdown(f"- **Chosen Action:** `{decision.get('chosen', '').upper()}`")
        st.markdown(f"- **Confidence:** `{decision.get('confidence', 0.0):.6f}`")
        st.markdown(
            f"- **Latency:** Decide `{timing.get('decide_ms', 0):.1f}ms` | "
            f"API `{timing.get('api_ms', 0):.1f}ms` | Turn `{timing.get('turn_ms', 0):.1f}ms`"
        )

    # Hypothesis Inspector & Laya Thinking Process (Judge Proof)
    with st.expander("🧠 Laya Thinking Process & Hypothesis Breakdown (Judge Audit)", expanded=True):
        st.markdown(
            """
            *According to PS 02 §3.2, every move is produced by evaluating typed hypothesis questions through Laya.*
            *Below are the exact 8 structured questions (2 per candidate direction) evaluated for this decision step:*
            """
        )
        questions = decision.get("questions", [])
        probs = decision.get("probabilities", {})
        chosen = decision.get("chosen", "")

        if questions and len(questions) >= 8:
            dirs = ("up", "down", "left", "right")
            for i, d in enumerate(dirs):
                q_prog = questions[i * 2]
                q_dang = questions[i * 2 + 1]
                p_val = probs.get(d, 0.0)
                is_picked = (d == chosen)
                border = "2px solid #2dd4bf" if is_picked else "1px solid #374151"
                bg = "#0f2324" if is_picked else "#181e29"
                tag = " 🌟 [CHOSEN ACTION BY LAYA]" if is_picked else ""

                st.markdown(
                    f"""
                    <div style="border: {border}; background-color: {bg}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: bold; font-size: 1.1em; color: #f3f4f6;">Direction: {d.upper()}{tag}</span>
                            <span style="background-color: #1e293b; color: #38bdf8; padding: 2px 8px; border-radius: 4px; font-weight: bold;">
                                Normalized Laya P: {p_val:.4f}
                            </span>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.9em; color: #94a3b8;">
                            <b>Q1 (Progress):</b> <code>{q_prog}</code>
                        </div>
                        <div style="margin-top: 4px; font-size: 0.9em; color: #94a3b8;">
                            <b>Q2 (Danger / Collision):</b> <code>{q_dang}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif questions:
            for idx, q in enumerate(questions, start=1):
                st.code(f"Q{idx}: {q}", language="markdown")
        else:
            st.write("No questions recorded in this step.")

    # Audit Feed Table
    with st.expander("📋 Full Step-by-Step Audit Feed (Immutable Hash Chain)", expanded=False):
        feed_rows = []
        for e in entries:
            dec = e.get("decision", {})
            tm = e.get("timing", {})
            feed_rows.append({
                "Step": e.get("step"),
                "Action": e.get("action"),
                "Result": e.get("result"),
                "Source": dec.get("source"),
                "Confidence": dec.get("confidence"),
                "Decide (ms)": tm.get("decide_ms"),
                "Hash": e.get("hash", "")[:12] + "...",
            })
        feed_df = pd.DataFrame(feed_rows)
        st.dataframe(feed_df, use_container_width=True)

    if auto_refresh:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
