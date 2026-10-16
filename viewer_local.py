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

st.set_page_config(
    page_title="ProgrammablePros · Laya EVA 02",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #080b12; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #0d1f3c 50%, #0a1628 100%);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-title { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin: 0; }
    .hero-sub   { font-size: 0.82rem; color: #64748b; margin-top: 4px; }
    .hero-badge {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white; font-size: 0.75rem; font-weight: 600;
        padding: 5px 14px; border-radius: 20px; letter-spacing: 0.05em;
    }
    .kpi-grid { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 110px;
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 14px 16px; text-align: center;
    }
    .kpi-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-top: 4px; line-height: 1; }
    .kpi-sub   { font-size: 0.7rem; color: #475569; margin-top: 3px; }
    .stage-pill { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; letter-spacing: 0.04em; margin-right: 8px; }
    .stage-locked   { background:#1e1b4b; color:#818cf8; border:1px solid #312e81; }
    .stage-active   { background:#042f2e; color:#2dd4bf; border:1px solid #0f766e; }
    .stage-complete { background:#052e16; color:#4ade80; border:1px solid #166534; }
    .section-title { font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
    .hash-pass { background: linear-gradient(135deg,#052e16,#042f2e); border: 1px solid #166534; border-radius: 10px; padding: 10px 14px; color: #4ade80; font-weight: 600; font-size: 0.85rem; }
    .hash-fail { background: #2d0a0a; border: 1px solid #7f1d1d; border-radius: 10px; padding: 10px 14px; color: #f87171; font-weight: 600; font-size: 0.85rem; }
    .dir-card { border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.06); background: #111827; }
    .dir-card-chosen { border: 1.5px solid #2dd4bf !important; background: linear-gradient(135deg,#042f2e,#0d2330) !important; }
    .dir-name { font-size: 1rem; font-weight: 700; color: #f1f5f9; }
    .dir-prob { font-size: 0.85rem; color: #38bdf8; font-weight: 600; font-family: 'JetBrains Mono',monospace; }
    .dir-q { font-size: 0.78rem; color: #94a3b8; margin-top: 5px; }
    .dir-q code { background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 0.74rem; }
    .chosen-tag { display:inline-block; background:linear-gradient(135deg,#0ea5e9,#2dd4bf); color:#000; font-size:0.65rem; font-weight:700; padding:2px 8px; border-radius:10px; margin-left:8px; letter-spacing:0.05em; vertical-align:middle; }
    .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
    .prob-label { width: 56px; font-size: 0.78rem; font-weight: 600; color: #cbd5e1; text-align: right; font-family: 'JetBrains Mono',monospace; }
    .prob-bar-wrap { flex: 1; height: 10px; background: #1e293b; border-radius: 5px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 5px; }
    .prob-val { width: 52px; font-size: 0.74rem; color: #64748b; font-family: 'JetBrains Mono',monospace; }
    .snake-grid-wrap { overflow-x: auto; }
    .snake-grid { display: inline-grid; gap: 2px; background: #0f172a; padding: 6px; border-radius: 8px; }
    .s-cell { width: 26px; height: 26px; border-radius: 3px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; }
    .s-empty { background: #131e2e; border: 1px solid rgba(255,255,255,0.03); }
    .s-head  { background: linear-gradient(135deg,#0ea5e9,#38bdf8); color: white; box-shadow: 0 0 6px rgba(14,165,233,0.6); }
    .s-body  { background: #0369a1; opacity: 0.85; }
    .s-food  { background: radial-gradient(circle, #f87171, #ef4444); box-shadow: 0 0 6px rgba(239,68,68,0.6); }
    .board-2048 { display: inline-grid; grid-template-columns: repeat(4, 82px); gap: 8px; background: #1e293b; padding: 10px; border-radius: 10px; }
    .tile-2048 { width: 82px; height: 82px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 800; }
    .fallback-bar-wrap { height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; margin-top: 5px; }
    .fallback-bar-ok   { height: 100%; background: linear-gradient(90deg,#4ade80,#22d3ee); border-radius: 4px; }
    .fallback-bar-warn { height: 100%; background: linear-gradient(90deg,#fb923c,#f87171); border-radius: 4px; }
    section[data-testid="stSidebar"] { background: #080b12; border-right: 1px solid rgba(255,255,255,0.05); }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

TILE_COLORS: dict[int, tuple[str, str]] = {
    0:    ("#1e293b", "#475569"),
    2:    ("#eee4da", "#776e65"),
    4:    ("#ede0c8", "#776e65"),
    8:    ("#f2b179", "#f9f6f2"),
    16:   ("#f59563", "#f9f6f2"),
    32:   ("#f67c5f", "#f9f6f2"),
    64:   ("#f65e3b", "#f9f6f2"),
    128:  ("#edcf72", "#f9f6f2"),
    256:  ("#edcc61", "#f9f6f2"),
    512:  ("#edc850", "#f9f6f2"),
    1024: ("#edc53f", "#f9f6f2"),
    2048: ("#edc22e", "#f9f6f2"),
}

DIRECTION_COLORS = {
    "up":    "#38bdf8",
    "down":  "#a78bfa",
    "left":  "#fb923c",
    "right": "#4ade80",
}


def load_log_entries(log_file: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not log_file.exists():
        return entries
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            try:
                entries.append(json.loads(ls))
            except json.JSONDecodeError:
                continue
    return entries


def collect_log_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("decision_log_*.jsonl",):
        files.extend(base.glob(pattern))
        files.extend((base / "snake").glob(pattern))
        files.extend((base / "2048").glob(pattern))
    return sorted(set(files), key=os.path.getmtime, reverse=True)


def _xy(seg: Any) -> tuple[int, int]:
    """Extract x, y from either a dict {x, y} or a list/tuple [x, y]."""
    if isinstance(seg, dict):
        return int(seg.get("x", 0)), int(seg.get("y", 0))
    return int(seg[0]), int(seg[1])


def render_snake_board(state: dict[str, Any]) -> None:
    width  = state.get("width",  20)
    height = state.get("height", 20)
    snake  = state.get("snake",  [])
    food   = state.get("food",   {"x": -1, "y": -1})
    grid: list[list[str]] = [["empty"] * width for _ in range(height)]
    for i, seg in enumerate(snake):
        x, y = _xy(seg)
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = "head" if i == 0 else "body"
    fx, fy = _xy(food)
    if 0 <= fx < width and 0 <= fy < height:
        grid[fy][fx] = "food"
    rows_html = ""
    for row in grid:
        cells = ""
        for cell in row:
            cls = {"empty": "s-empty", "head": "s-head", "body": "s-body", "food": "s-food"}[cell]
            sym = {"empty": "", "head": "", "body": "", "food": ""}[cell]
            cells += f'<div class="s-cell {cls}">{sym}</div>'
        rows_html += cells
    cols_css = f"repeat({width}, 26px)"
    st.markdown(
        f"""<div class="snake-grid-wrap">
          <div class="snake-grid" style="grid-template-columns:{cols_css}">{rows_html}</div>
        </div>
        <div style="margin-top:8px;font-size:0.72rem;color:#475569;">
          <span style="background:#0ea5e9;padding:1px 6px;border-radius:3px;color:#fff;margin-right:6px;">Head</span>
          <span style="background:#0369a1;padding:1px 6px;border-radius:3px;color:#fff;margin-right:6px;">Body</span>
          <span style="background:#ef4444;padding:1px 6px;border-radius:3px;color:#fff;">? Food</span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_2048_board(state: dict[str, Any]) -> None:
    board = state.get("board", [[0] * 4 for _ in range(4)])
    tiles_html = ""
    for row in board:
        for val in row:
            bg, fg = TILE_COLORS.get(val, ("#3c3a32", "#f9f6f2"))
            label  = str(val) if val > 0 else ""
            fsize  = "22px" if val < 1000 else "16px"
            tiles_html += f'<div class="tile-2048" style="background:{bg};color:{fg};font-size:{fsize};">{label}</div>'
    st.markdown(f'<div class="board-2048">{tiles_html}</div>', unsafe_allow_html=True)


def render_prob_bars(probs: dict[str, float], chosen: str) -> None:
    if not probs:
        st.caption("No probability data.")
        return
    html = ""
    for direction in ("up", "down", "left", "right"):
        val   = probs.get(direction, 0.0)
        color = DIRECTION_COLORS.get(direction, "#94a3b8")
        pct   = f"{val * 100:.1f}%"
        star  = " ?" if direction == chosen else ""
        html += f"""<div class="prob-row">
          <div class="prob-label" style="color:{color};">{direction.upper()}{star}</div>
          <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{pct};background:{color};opacity:0.85;"></div></div>
          <div class="prob-val">{val:.4f}</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(entries: list[dict], latest: dict, state: dict, game: str) -> None:
    session_short = latest.get("session_id", "N/A")
    session_short = session_short[:8] + "…" if len(session_short) > 8 else session_short
    step     = latest.get("step", 0)
    score    = state.get("score", 0)
    action   = latest.get("action", "—").upper()
    applied  = sum(1 for e in entries if e.get("result") != "rejected")
    rejected = len(entries) - applied
    if game == "snake":
        food_eaten = state.get("foodEaten", state.get("foods_eaten", 0)) or 0
        won = state.get("won", False) or state.get("game_over", False) and food_eaten >= 15
        real_score = food_eaten * 60 + (100 if won else 0)
        extra_label, extra_val, extra_sub = "Foods", f"{food_eaten}/15", "target"
        score = real_score
    else:
        extra_label, extra_val, extra_sub = "Best Tile", str(state.get("best_tile", 0)), "target 512+"
    st.markdown(
        f"""<div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Session</div><div class="kpi-value" style="font-size:1rem;">{session_short}</div></div>
          <div class="kpi-card"><div class="kpi-label">Step</div><div class="kpi-value">{step}</div><div class="kpi-sub">{applied} applied · {rejected} rejected</div></div>
          <div class="kpi-card"><div class="kpi-label">Score</div><div class="kpi-value" style="color:#38bdf8;">{score}</div></div>
          <div class="kpi-card"><div class="kpi-label">{extra_label}</div><div class="kpi-value" style="color:#4ade80;">{extra_val}</div><div class="kpi-sub">{extra_sub}</div></div>
          <div class="kpi-card"><div class="kpi-label">Last Move</div><div class="kpi-value" style="font-size:1.1rem;color:#a78bfa;">{action}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_status_strip(entries: list[dict], log_file: Path, decision: dict) -> None:
    fb_count = sum(1 for e in entries if e.get("decision", {}).get("source") == "fallback")
    fb_pct   = (fb_count / len(entries) * 100.0) if entries else 0.0
    is_valid, _ = OfficialLogger.verify_log_file(log_file)
    src       = decision.get("source", "laya")
    src_color = "#2dd4bf" if src == "laya" else "#facc15"
    src_bg    = "#042f2e" if src == "laya" else "#422006"
    bar_cls   = "fallback-bar-ok" if fb_pct <= 10 else "fallback-bar-warn"
    bar_pct   = f"{min(fb_pct, 100):.1f}%"
    hash_html = (
        '<div class="hash-pass">? &nbsp;Hash Chain VALID — SHA-256 audit intact</div>'
        if is_valid else
        '<div class="hash-fail">? &nbsp;Hash Chain INVALID — integrity compromised</div>'
    )
    st.markdown(
        f"""<div style="display:flex;gap:14px;margin-bottom:16px;flex-wrap:wrap;">
          <div style="flex:1;min-width:160px;background:{src_bg};border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;">
            <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Decision Source</div>
            <div style="font-size:1rem;font-weight:700;color:{src_color};margin-top:3px;">{src.upper()}</div>
          </div>
          <div style="flex:1;min-width:160px;background:#0f172a;border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;">
            <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Fallback Rate</div>
            <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-top:3px;">{fb_pct:.1f}% <span style="font-size:0.7rem;color:#475569;">/ 10% budget</span></div>
            <div class="fallback-bar-wrap"><div class="{bar_cls}" style="width:{bar_pct};"></div></div>
          </div>
          <div style="flex:2;min-width:240px;">{hash_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_hypothesis_inspector(decision: dict) -> None:
    questions = decision.get("questions", [])
    probs     = decision.get("probabilities", {})
    chosen    = decision.get("chosen", "")
    if not questions:
        st.caption("No hypothesis data recorded in this step.")
        return
    dirs = ("up", "down", "left", "right")
    html = ""
    for i, d in enumerate(dirs):
        q_prog = questions[i * 2]     if len(questions) > i * 2     else "—"
        q_dang = questions[i * 2 + 1] if len(questions) > i * 2 + 1 else "—"
        p_val  = probs.get(d, 0.0)
        color  = DIRECTION_COLORS.get(d, "#94a3b8")
        is_chosen = d == chosen
        card_cls  = "dir-card dir-card-chosen" if is_chosen else "dir-card"
        tag_html  = '<span class="chosen-tag">? CHOSEN</span>' if is_chosen else ""
        html += f"""<div class="{card_cls}">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="dir-name" style="color:{color};">{d.upper()}{tag_html}</span>
            <span class="dir-prob">P = {p_val:.4f}</span>
          </div>
          <div class="dir-q"><b>Q1 Progress:</b> <code>{q_prog}</code></div>
          <div class="dir-q"><b>Q2 Danger:</b>   <code>{q_dang}</code></div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_audit_table(entries: list[dict]) -> None:
    rows = []
    for e in entries:
        dec = e.get("decision", {})
        tm  = e.get("timing",   {})
        rows.append({
            "Step":       e.get("step"),
            "Action":     (e.get("action") or "").upper(),
            "Result":     e.get("result", ""),
            "Source":     dec.get("source", ""),
            "Confidence": f"{dec.get('confidence', 0.0):.4f}",
            "Decide ms":  f"{tm.get('decide_ms', 0):.1f}",
            "Hash":       (e.get("hash", "")[:14] + "…") if e.get("hash") else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)


def main() -> None:
    st.markdown(
        """<div class="hero-header">
          <div>
            <div class="hero-title">&#129302; ProgrammablePros &mdash; Laya Decision Engine</div>
            <div class="hero-sub">Genesis 2.0 · Track EVA 02 · Live Telemetry &amp; Cryptographic Audit</div>
          </div>
          <div class="hero-badge">HACKATHON LIVE</div>
        </div>""",
        unsafe_allow_html=True,
    )

    log_dir   = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    all_files = collect_log_files(log_dir)

    snake_best, tile_best = 0, 0
    for f in all_files:
        entries_q = load_log_entries(f)
        if not entries_q:
            continue
        last    = entries_q[-1]
        state_q = last.get("state_after", last.get("state_before", {}))
        game_q  = last.get("game", "snake")
        if game_q == "snake":
            fe = state_q.get("foodEaten", state_q.get("foods_eaten", 0)) or 0
            won_q = state_q.get("won", False) or (state_q.get("game_over", False) and fe >= 15)
            snake_best = max(snake_best, fe * 60 + (100 if won_q else 0))
        else:
            tile_best = max(tile_best, state_q.get("best_tile", 0))

    s1_class = "stage-complete" if snake_best >= 600 else "stage-active"
    s1_label = f"\u2705 Stage 1 Snake \u2014 Best {snake_best} pts" if snake_best >= 600 else f"\U0001f40d Stage 1 Snake \u2014 Best {snake_best} pts (need 600)"
    s2_class = "stage-complete" if tile_best >= 512 else ("stage-active" if snake_best >= 600 else "stage-locked")
    s2_label = (f"\u2705 Stage 2 2048 \u2014 Best tile {tile_best}" if tile_best >= 512
                else (f"\U0001f7e2 Stage 2 2048 \u2014 Best tile {tile_best} (need 512)" if snake_best >= 600
                      else "\U0001f512 Stage 2 2048 \u2014 Locked (complete Stage 1 first)"))
    st.markdown(
        f"""<div style="margin-bottom:18px;">
          <span class="stage-pill {s1_class}">{s1_label}</span>
          <span class="stage-pill {s2_class}">{s2_label}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:14px;'>⚙️ Session Inspector</div>", unsafe_allow_html=True)
        if not all_files:
            st.warning("No log files found in logs/ directory.")
            st.info("Run a game to generate a session log.")
            return
        selected_file = st.selectbox(
            "Log file",
            all_files,
            format_func=lambda p: f"{'\U0001f40d' if 'snake' in str(p) else '\U0001f7e2'} {p.name}",
        )
        auto_refresh = st.checkbox("? Auto-refresh (1s)", value=True)
        st.markdown("---")
        st.caption(f"📁 {selected_file.parent.name}/{selected_file.name}")
        st.caption(f"📦 {selected_file.stat().st_size // 1024} KB")

    entries = load_log_entries(selected_file)
    if not entries:
        st.warning(f"Log `{selected_file.name}` is empty — waiting for game data…")
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        return

    latest   = entries[-1]
    game     = latest.get("game", "snake")
    state    = latest.get("state_after", latest.get("state_before", {}))
    decision = latest.get("decision", {})
    timing   = latest.get("timing",   {})

    render_kpi_row(entries, latest, state, game)
    render_status_strip(entries, selected_file, decision)

    col_board, col_right = st.columns([1.15, 0.85], gap="medium")

    with col_board:
        game_icon = "🐍" if game == "snake" else "🔢"
        st.markdown(f'<div class="section-title">{game_icon} {"Snake Board (20×20)" if game == "snake" else "2048 Board (4×4)"}</div>', unsafe_allow_html=True)
        if game == "snake":
            render_snake_board(state)
            head_raw = state.get("snake", [{"x": 0, "y": 0}])[0]
            food_raw = state.get("food",  {"x": 0, "y": 0})
            hx, hy   = _xy(head_raw)
            fx2, fy2 = _xy(food_raw)
            direc    = state.get("direction", "—")
            length   = len(state.get("snake", []))
            st.markdown(
                f"""<div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;">
                  <div style="background:#1e293b;border-radius:8px;padding:7px 12px;font-size:0.78rem;color:#94a3b8;">🎯 Head <b style="color:#f1f5f9;">[{hx},{hy}]</b></div>
                  <div style="background:#1e293b;border-radius:8px;padding:7px 12px;font-size:0.78rem;color:#94a3b8;">🍎 Food <b style="color:#f87171;">[{fx2},{fy2}]</b></div>
                  <div style="background:#1e293b;border-radius:8px;padding:7px 12px;font-size:0.78rem;color:#94a3b8;">📏 Length <b style="color:#f1f5f9;">{length}</b></div>
                  <div style="background:#1e293b;border-radius:8px;padding:7px 12px;font-size:0.78rem;color:#94a3b8;">➡️ Dir <b style="color:#a78bfa;">{direc.upper()}</b></div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            render_2048_board(state)

    with col_right:
        probs  = decision.get("probabilities", {})
        chosen = decision.get("chosen", "")
        st.markdown('<div class="section-title">&#128202; Calibrated Probabilities</div>', unsafe_allow_html=True)
        render_prob_bars(probs, chosen)
        st.markdown("<br>", unsafe_allow_html=True)

        confidence = decision.get("confidence", 0.0)
        decide_ms  = timing.get("decide_ms", 0)
        api_ms     = timing.get("api_ms",    0)
        turn_ms    = timing.get("turn_ms",   0)
        st.markdown('<div class="section-title">? Performance</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div style="display:flex;flex-direction:column;gap:7px;">
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:#64748b;">Confidence</span>
                <span style="color:#38bdf8;font-family:'JetBrains Mono',monospace;font-weight:600;">{confidence:.6f}</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:#64748b;">Decide latency</span>
                <span style="color:#f1f5f9;font-family:'JetBrains Mono',monospace;">{decide_ms:.1f} ms</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:#64748b;">API latency</span>
                <span style="color:#f1f5f9;font-family:'JetBrains Mono',monospace;">{api_ms:.1f} ms</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:#64748b;">Total turn time</span>
                <span style="color:#f1f5f9;font-family:'JetBrains Mono',monospace;">{turn_ms:.1f} ms</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🧠 Laya Hypothesis Inspector — Typed Questions & EV Breakdown", expanded=True):
        st.caption("Per §3.2: every move is produced by evaluating structured hypothesis questions through Laya. Shown below are the typed questions (2 per direction) evaluated for this decision step.")
        render_hypothesis_inspector(decision)

    with st.expander("📋 Full Step-by-Step Audit Feed (SHA-256 Hash Chain)", expanded=False):
        render_audit_table(entries)

    if auto_refresh:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
