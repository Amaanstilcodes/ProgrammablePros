"""Streamlit Live Decision Viewer Module.

Reads decision_log.jsonl in real-time and visualizes:
- Current game state & progress toward goal
- Calibrated probability distributions returned by Laya
- Formulated questions sent to Laya
- Chronological decision audit history
"""

import time
from typing import Any, Dict, List

import streamlit as st

from logger import DecisionLogger


def setup_ui() -> None:
    """Configure Streamlit page layout and custom styling."""
    st.set_page_config(
        page_title="Laya Bot Live Viewer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling for clean hackathon dashboard appearance
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #1e222d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid #4CAF50;
        }
        .action-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .action-right {
            background-color: #1b5e20;
            color: #a5d6a7;
        }
        .action-left {
            background-color: #b71c1c;
            color: #ffcdd2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit application entry point."""
    setup_ui()

    # Sidebar configuration
    st.sidebar.title("⚙️ Dashboard Controls")
    log_file_path = st.sidebar.text_input("Log File Path", value="decision_log.jsonl")
    auto_refresh = st.sidebar.checkbox("Live Auto-Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    max_history = st.sidebar.slider("Show Recent Decisions", min_value=5, max_value=25, value=10)

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        ### 🧠 Architecture
        - **Model:** Laya (Calibrated Probability)
        - **Pipeline:**
          1. State observation
          2. Typed query formulation
          3. Probability scoring
          4. Argmax action execution
        """
    )

    if st.sidebar.button("🔄 Manual Refresh"):
        st.rerun()

    # Main dashboard header
    st.title("🤖 Laya Game Bot — Live Decision Viewer")
    st.caption("Real-time telemetry and decision probability calibration stream")

    # Read records using DecisionLogger
    logger = DecisionLogger(log_path=log_file_path)
    records: List[Dict[str, Any]] = logger.read_recent_logs(limit=None)

    if not records:
        st.info(
            "⏳ **Waiting for decision logs...**\n\n"
            "Run `python main.py` in your terminal to start the game loop and populate decisions."
        )
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
        return

    latest = records[-1]
    latest_state = latest.get("game_state", {})
    latest_probs = latest.get("probabilities_returned", {})
    latest_evs = latest.get("expected_values", {})
    latest_ev = latest.get("expected_value")
    latest_action = latest.get("chosen_action", "N/A")
    latest_compressed = latest.get("compressed_state", "N/A")
    latest_warning = latest.get("adaptive_warning")
    player_pos = latest_state.get("player_pos", 0)
    goal_pos = latest_state.get("goal_pos", 10)
    game_status = latest_state.get("status", "IN_PROGRESS")

    # Top Level KPI Metrics (5 columns)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Decisions Logged", len(records))
    with col2:
        st.metric("Player Position", f"{player_pos} / {goal_pos}")
    with col3:
        st.metric("Latest Chosen Move", latest_action.upper())
    with col4:
        ev_display = f"{latest_ev:+.2f}" if latest_ev is not None else "N/A"
        st.metric("Expected Value (EV)", ev_display)
    with col5:
        status_color = "🟢" if game_status == "WON" else ("🟡" if game_status == "IN_PROGRESS" else "🔴")
        st.metric("Game Status", f"{status_color} {game_status}")

    # Feature 4 Alert: Adaptive Warning Callout
    if latest_warning:
        st.warning(f"⚠️ **Failure-Adaptive Warning Triggered:** {latest_warning}")

    # Feature 1 Context: Compressed State Display
    st.info(f"🧬 **Feature 1 — Compressed State Context:** `{latest_compressed}`")

    # Goal Progress Bar
    progress_val = min(1.0, max(0.0, player_pos / goal_pos if goal_pos > 0 else 0.0))
    st.progress(progress_val, text=f"Progress to Goal: {player_pos}/{goal_pos} ({progress_val:.0%})")

    st.divider()

    # Active Decision Inspection
    st.subheader("🎯 Latest Evaluated Move (Probabilities vs Expected Values)")
    col_probs, col_context = st.columns([1, 1])

    with col_probs:
        st.markdown("**Calibrated Probabilities & EV Scores:**")
        for action, prob in latest_probs.items():
            ev_score = latest_evs.get(action)
            ev_str = f" | EV: **{ev_score:+.2f}**" if ev_score is not None else ""
            st.write(f"**Action `{action}`**: {prob:.1%}{ev_str}")
            st.progress(min(1.0, max(0.0, float(prob))))

    with col_context:
        st.markdown("**Typed Hypothesis Questions Evaluated by Laya:**")
        questions = latest.get("questions_asked", {})
        if isinstance(questions, dict):
            for act, q in questions.items():
                st.markdown(f"- **`{act}`**: *\"{q}\"*")
        else:
            st.write(questions)

        with st.expander("🔍 View Raw State JSON"):
            st.json(latest_state)

    st.divider()

    # Historical Decisions Feed
    st.subheader(f"📜 Recent Decision History (Latest {min(len(records), max_history)})")
    recent_records = list(reversed(records[-max_history:]))

    for idx, item in enumerate(recent_records):
        step_num = len(records) - idx
        action = item.get("chosen_action", "N/A")
        ts = item.get("timestamp", "")
        probs = item.get("probabilities_returned", {})
        evs = item.get("expected_values", {})
        chosen_ev_item = item.get("expected_value")
        warning_item = item.get("adaptive_warning")
        compressed_item = item.get("compressed_state", "N/A")
        state = item.get("game_state", {})

        prob_summary = " | ".join(f"{a}: {p:.1%}" for a, p in probs.items())
        ev_summary = " | ".join(f"{a}: {e:+.2f}" for a, e in evs.items()) if evs else ""
        badge_class = "action-right" if action.lower() == "right" else "action-left"
        warning_badge = " ⚠️ [ADAPTED]" if warning_item else ""

        with st.expander(f"Step {step_num:02d} — Action: [{action.upper()}]{warning_badge} — EV: {chosen_ev_item:+.2f} — {ts}"):
            st.markdown(
                f"<span class='action-badge {badge_class}'>Action: {action}</span> "
                f"&nbsp;&nbsp; **EV:** `{chosen_ev_item:+.2f}` "
                f"&nbsp;&nbsp; **Probabilities:** `{prob_summary}`",
                unsafe_allow_html=True,
            )
            if ev_summary:
                st.caption(f"Action Expected Values: {ev_summary}")
            st.markdown(f"**Compressed State:** `{compressed_item}`")
            st.markdown(f"**Position at Decision:** `{state.get('player_pos', 'N/A')}` / `{state.get('goal_pos', 'N/A')}`")

            if warning_item:
                st.markdown(f"**Memory Warning Active:** `{warning_item}`")

            col_q, col_raw = st.columns(2)
            with col_q:
                st.caption("Typed Questions Evaluated:")
                st.write(item.get("questions_asked", {}))
            with col_raw:
                st.caption("Raw State Record:")
                st.json(state)

    # Auto-refresh loop
    if auto_refresh and game_status != "WON":
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
