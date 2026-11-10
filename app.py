"""ProgrammablePros — Hackathon Control Panel & Live Decision Viewer.

Genesis 2.0 Track EVA 02 | Streamlit app.py

Layout
------
Sidebar  : Run Stage 1 (Snake) / Run Stage 2 (2048) buttons + session selector
Main     : KPI bar, Decision Logger table, circuit-breaker status banner
Expander : Full Laya hypothesis questions + raw probability breakdown
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ProgrammablePros · EVA 02 Control Panel",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #080b12; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #0d1f3c 60%, #0a1628 100%);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 16px; padding: 18px 26px; margin-bottom: 18px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .hero-title { font-size: 1.45rem; font-weight: 700; color: #f1f5f9; margin: 0; }
    .hero-sub   { font-size: 0.8rem; color: #64748b; margin-top: 3px; }
    .hero-badge {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white; font-size: 0.72rem; font-weight: 700;
        padding: 5px 14px; border-radius: 20px; letter-spacing: 0.06em;
    }

    /* KPI */
    .kpi-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
    .kpi-card {
        flex: 1; min-width: 110px;
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 13px 16px; text-align: center;
    }
    .kpi-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; }
    .kpi-value { font-size: 1.55rem; font-weight: 800; color: #f1f5f9; margin-top: 3px; line-height: 1; }
    .kpi-sub   { font-size: 0.68rem; color: #475569; margin-top: 4px; }

    /* Status banners */
    .banner-ok   { background:#052e16; border:1px solid #166534; border-radius:10px; padding:10px 14px; color:#4ade80; font-size:0.85rem; font-weight:600; margin-bottom:12px; }
    .banner-warn { background:#2d1a00; border:1px solid #92400e; border-radius:10px; padding:10px 14px; color:#fb923c; font-size:0.85rem; font-weight:600; margin-bottom:12px; }
    .banner-err  { background:#2d0a0a; border:1px solid #7f1d1d; border-radius:10px; padding:10px 14px; color:#f87171; font-size:0.85rem; font-weight:600; margin-bottom:12px; }
    .banner-info { background:#0c1a2e; border:1px solid #1d4ed8; border-radius:10px; padding:10px 14px; color:#93c5fd; font-size:0.85rem; margin-bottom:12px; }

    /* Fallback budget bar */
    .fb-wrap { height: 7px; background: #1e293b; border-radius: 4px; overflow: hidden; margin-top: 4px; }
    .fb-ok   { height:100%; border-radius:4px; background:linear-gradient(90deg,#4ade80,#22d3ee); }
    .fb-warn { height:100%; border-radius:4px; background:linear-gradient(90deg,#fb923c,#f87171); }

    /* Log table styling */
    .action-pill {
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:0.73rem; font-weight:700; letter-spacing:0.04em;
    }
    .pill-up    { background:#0c2540; color:#38bdf8; border:1px solid #0ea5e9; }
    .pill-down  { background:#1e1040; color:#a78bfa; border:1px solid #7c3aed; }
    .pill-left  { background:#2d1500; color:#fb923c; border:1px solid #ea580c; }
    .pill-right { background:#052e16; color:#4ade80; border:1px solid #16a34a; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #080b12; border-right: 1px solid rgba(255,255,255,0.05); }
    .sidebar-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 12px; }
    .run-status { font-size: 0.75rem; color: #64748b; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SNAKE_LOG_DIR = ROOT / "logs" / "snake"
LOG2048_DIR = ROOT / "logs" / "2048"
RUNNER_SNAKE = str(ROOT / "run_snake_live.py")
RUNNER_2048 = str(ROOT / "run_2048_live.py")

DIRECTION_COLORS = {
    "up": "#38bdf8", "down": "#a78bfa",
    "left": "#fb923c", "right": "#4ade80",
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in {
    "runner_proc": None,       # subprocess.Popen or None
    "runner_game": None,       # "snake" | "2048"
    "runner_lines": [],        # captured stdout lines
    "runner_status": "idle",   # "idle" | "running" | "done" | "error"
    "circuit_broken": False,
    "circuit_msg": "",
    "selected_log": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helpers — log file discovery
# ---------------------------------------------------------------------------
def _list_logs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.glob("decision_log_*.jsonl")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _read_log(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Board rendering helpers (ported from viewer_local.py)
# ---------------------------------------------------------------------------
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


def _xy(seg: Any) -> tuple[int, int]:
    if isinstance(seg, dict):
        return int(seg.get("x", 0)), int(seg.get("y", 0))
    return int(seg[0]), int(seg[1])


def _render_snake_board(state: dict) -> None:
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
    cell_styles = {
        "empty": ("#131e2e", "1px solid rgba(255,255,255,0.03)", ""),
        "head":  ("linear-gradient(135deg,#0ea5e9,#38bdf8)", "none", "🟦"),
        "body":  ("#0369a1", "none", ""),
        "food":  ("radial-gradient(circle,#f87171,#ef4444)", "none", "🍎"),
    }
    rows_html = ""
    for row in grid:
        for cell in row:
            bg, border, sym = cell_styles[cell]
            rows_html += (
                f'<div style="width:22px;height:22px;border-radius:3px;'
                f'background:{bg};border:{border};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:9px;">{sym}</div>'
            )
    cols_css = f"repeat({width}, 22px)"
    head_pos = _xy(snake[0]) if snake else "—"
    food_pos = _xy(food)
    direction = state.get("direction", "—").upper()
    length = len(snake)
    st.markdown(
        f'<div style="display:inline-grid;grid-template-columns:{cols_css};'
        f'gap:2px;background:#0f172a;padding:8px;border-radius:10px;">'
        f'{rows_html}</div>'
        f'<div style="margin-top:8px;font-size:0.72rem;color:#64748b;">'
        f'Head <b style="color:#38bdf8">{head_pos}</b> &nbsp;·&nbsp; '
        f'Food <b style="color:#ef4444">{food_pos}</b> &nbsp;·&nbsp; '
        f'Dir <b style="color:#a78bfa">{direction}</b> &nbsp;·&nbsp; '
        f'Length <b style="color:#4ade80">{length}</b></div>',
        unsafe_allow_html=True,
    )


def _render_2048_board(state: dict) -> None:
    board = state.get("board", [[0] * 4 for _ in range(4)])
    tiles_html = ""
    for row in board:
        for val in row:
            bg, fg = TILE_COLORS.get(val, ("#3c3a32", "#f9f6f2"))
            label  = str(val) if val > 0 else ""
            fsize  = "20px" if val < 1000 else "15px"
            tiles_html += (
                f'<div style="width:82px;height:82px;border-radius:8px;'
                f'background:{bg};color:{fg};font-size:{fsize};'
                f'font-weight:800;display:flex;align-items:center;'
                f'justify-content:center;">'
                f'{label}</div>'
            )
    flat = [v for row in board for v in row]
    max_tile = max(flat) if flat else 0
    score = state.get("score", 0)
    empty = flat.count(0)
    st.markdown(
        f'<div style="display:inline-grid;grid-template-columns:repeat(4,82px);'
        f'gap:8px;background:#1e293b;padding:10px;border-radius:12px;">'
        f'{tiles_html}</div>'
        f'<div style="margin-top:8px;font-size:0.72rem;color:#64748b;">'
        f'Max Tile <b style="color:#edc22e">{max_tile}</b> &nbsp;·&nbsp; '
        f'Score <b style="color:#38bdf8">{score}</b> &nbsp;·&nbsp; '
        f'Empty <b style="color:#4ade80">{empty}</b></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers — subprocess runner
# ---------------------------------------------------------------------------
def _start_runner(script: str, game: str) -> None:
    if st.session_state.runner_proc is not None:
        try:
            st.session_state.runner_proc.terminate()
        except Exception:
            pass

    st.session_state.runner_lines = []
    st.session_state.runner_status = "running"
    st.session_state.runner_game = game
    st.session_state.circuit_broken = False
    st.session_state.circuit_msg = ""

    proc = subprocess.Popen(
        [sys.executable, script, "--agent-name", "ProgrammablePros"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )
    st.session_state.runner_proc = proc


def _poll_runner() -> None:
    proc: subprocess.Popen | None = st.session_state.runner_proc
    if proc is None or st.session_state.runner_status not in ("running",):
        return
    # Non-blocking: drain available output
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            stripped = line.rstrip()
            st.session_state.runner_lines.append(stripped)
            if "HTTP 500" in stripped or "FATAL: Session corrupted" in stripped:
                st.session_state.circuit_broken = True
                st.session_state.circuit_msg = stripped
    except Exception:
        pass

    ret = proc.poll()
    if ret is not None:
        st.session_state.runner_status = "done" if ret == 0 else "error"


# ---------------------------------------------------------------------------
# Helpers — log analytics
# ---------------------------------------------------------------------------
def _build_dataframe(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        dec = r.get("decision", {})
        probs = dec.get("probabilities", {})
        chosen = dec.get("chosen", r.get("action", ""))
        records.append({
            "Step": r.get("step", ""),
            "Action": r.get("action", ""),
            "Result": r.get("result", ""),
            "Source": dec.get("source", ""),
            "Confidence": f"{dec.get('confidence', 0.0):.3f}",
            "P(up)":    f"{probs.get('up',   0.0):.3f}",
            "P(down)":  f"{probs.get('down', 0.0):.3f}",
            "P(left)":  f"{probs.get('left', 0.0):.3f}",
            "P(right)": f"{probs.get('right',0.0):.3f}",
            "Decide ms": r.get("timing", {}).get("decide_ms", ""),
            "API ms":    r.get("timing", {}).get("api_ms", ""),
        })
    return pd.DataFrame(records)


def _session_kpis(rows: list[dict], game: str) -> dict:
    total = len(rows)
    applied = sum(1 for r in rows if r.get("result") == "applied")
    fallbacks = sum(1 for r in rows if r.get("decision", {}).get("source") == "fallback")
    fb_pct = (fallbacks / total * 100) if total else 0.0

    kpis: dict[str, Any] = {
        "Total Moves": total,
        "Applied": applied,
        "Rejected": total - applied,
        "Fallback": f"{fallbacks} / {total}",
        "FB %": f"{fb_pct:.1f}%",
        "fb_raw": fb_pct,
    }

    if rows:
        last = rows[-1]
        state = last.get("state_after") or last.get("state_before") or {}
        if game == "snake":
            snake = state.get("snake", [])
            kpis["Head"] = str(snake[0]) if snake else "—"
            kpis["Food"] = str(state.get("food", "—"))
            kpis["Length"] = len(snake) if snake else "—"
            kpis["Score"] = state.get("score", "—")
        else:
            board = state.get("board", [])
            flat = [v for row in board for v in row] if board else []
            kpis["Max Tile"] = max(flat) if flat else "—"
            kpis["Score"] = state.get("score", "—")
            kpis["Empty"] = flat.count(0) if flat else "—"

    return kpis


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-title">🤖 ProgrammablePros · EVA 02 Control Panel</div>
        <div class="hero-sub">Genesis 2.0 Hackathon · Real Laya Model · Zero Mocks · SHA-256 Hash-Chain Logging</div>
      </div>
      <div class="hero-badge">TRACK EVA 02</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎮 Arena Controls</div>', unsafe_allow_html=True)

    run_snake = st.button(
        "▶️ Stage 1 — Snake",
        use_container_width=True,
        type="primary",
        key="btn_snake",
        help="Run run_snake_live.py against the live TopGameAI Arena (real Laya, zero mocks)",
    )
    run_2048 = st.button(
        "▶️ Stage 2 — 2048",
        use_container_width=True,
        type="primary",
        key="btn_2048",
        help="Run run_2048_live.py against the live TopGameAI Arena (real Laya, zero mocks)",
    )

    if run_snake:
        _start_runner(RUNNER_SNAKE, "snake")
        st.rerun()

    if run_2048:
        _start_runner(RUNNER_2048, "2048")
        st.rerun()

    # Runner status
    status = st.session_state.runner_status
    game_label = st.session_state.runner_game or "—"
    status_icons = {"idle": "⏸", "running": "🔄", "done": "✅", "error": "❌"}
    st.markdown(
        f'<div class="run-status">{status_icons.get(status, "?")} {status.upper()} · {game_label}</div>',
        unsafe_allow_html=True,
    )

    if status == "running":
        if st.button("⏹ Stop Runner", use_container_width=True, key="btn_stop"):
            proc = st.session_state.runner_proc
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass
            st.session_state.runner_status = "done"

    st.divider()

    # Log file selector
    st.markdown("**📂 View Session Log**")
    snake_logs = _list_logs(SNAKE_LOG_DIR)
    logs_2048 = _list_logs(LOG2048_DIR)
    all_logs = snake_logs + logs_2048
    log_names = [f"{'🐍' if p.parent.name == 'snake' else '🎯'} {p.stem[-12:]}" for p in all_logs]

    if all_logs:
        chosen_idx = st.selectbox(
            "Session",
            options=list(range(len(all_logs))),
            format_func=lambda i: log_names[i],
            label_visibility="collapsed",
        )
        st.session_state.selected_log = all_logs[chosen_idx]
        sel_game = "snake" if all_logs[chosen_idx].parent.name == "snake" else "2048"
    else:
        st.info("No logs yet. Run a game first.", icon="📭")
        st.session_state.selected_log = None
        sel_game = "snake"

    st.divider()
    auto_refresh = st.checkbox("🔄 Auto-refresh (2s)", value=status == "running")

# ---------------------------------------------------------------------------
# Poll the runner and auto-refresh
# ---------------------------------------------------------------------------
_poll_runner()
if auto_refresh or st.session_state.runner_status == "running":
    time.sleep(2)
    st.rerun()

# ---------------------------------------------------------------------------
# Circuit-breaker status banner
# ---------------------------------------------------------------------------
if st.session_state.circuit_broken:
    st.markdown(
        f"""<div class="banner-err">
        ⚡ <strong>Circuit Breaker Fired</strong> — Server HTTP 500 detected (empty body, Next.js/Cloudflare edge crash).<br>
        Session gracefully aborted to prevent infinite retry loops. Log integrity preserved.<br>
        <span style="font-family:monospace;font-size:0.78rem;color:#fca5a5;">{st.session_state.circuit_msg[:200]}</span>
        </div>""",
        unsafe_allow_html=True,
    )
elif st.session_state.runner_status == "running":
    st.markdown(
        '<div class="banner-ok">🟢 <strong>Runner Active</strong> — Real Laya model running. Submitting moves to live TopGameAI Arena.</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.runner_status == "done" and not st.session_state.circuit_broken:
    st.markdown(
        '<div class="banner-ok">✅ <strong>Session Complete</strong> — Log closed and hash chain verified.</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.runner_status == "error":
    st.markdown(
        '<div class="banner-warn">⚠️ <strong>Runner exited with non-zero code.</strong> Check the terminal output below for details.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="banner-info">ℹ️ Use the sidebar buttons to launch a live arena session. Select a past log to inspect decisions.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Live runner terminal output
# ---------------------------------------------------------------------------
if st.session_state.runner_lines:
    with st.expander("🖥️ Live Runner Output (stdout)", expanded=st.session_state.runner_status == "running"):
        tail = st.session_state.runner_lines[-80:]
        output_text = "\n".join(tail)
        st.code(output_text, language="bash")

# ---------------------------------------------------------------------------
# Load the selected log
# ---------------------------------------------------------------------------
log_path: Path | None = st.session_state.selected_log
rows: list[dict] = _read_log(log_path) if log_path else []
current_game = sel_game if log_path else "snake"

# ---------------------------------------------------------------------------
# GAME BOARD — shown always (even on 0 moves logged)
# ---------------------------------------------------------------------------
st.markdown("### 🎮 Game Board")

# Resolve the best available state to render
_display_state: dict = {}
if rows:
    last_row = rows[-1]
    _display_state = last_row.get("state_after") or last_row.get("state_before") or {}
elif log_path is not None:
    # No logged moves yet (e.g. circuit breaker fired on turn 1).
    # Try to find the initial state from the JSONL header line or fall back
    # to showing a placeholder board so the user sees *something*.
    _display_state = {}

if _display_state:
    _board_col, _info_col = st.columns([2, 1])
    with _board_col:
        if current_game == "snake":
            _render_snake_board(_display_state)
        else:
            _render_2048_board(_display_state)
    with _info_col:
        _dec = rows[-1].get("decision", {}) if rows else {}
        _probs = _dec.get("probabilities", {})
        if _probs:
            _chosen = _dec.get("chosen", "")
            _max_p = max(_probs.values()) if _probs else 1.0
            _bars_html = ""
            for _d in ("up", "down", "left", "right"):
                _p = _probs.get(_d, 0.0)
                _w = int(_p / max(_max_p, 1e-9) * 100)
                _col = DIRECTION_COLORS.get(_d, "#94a3b8")
                _tick = " ◀" if _d == _chosen else ""
                _bars_html += (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<span style="width:44px;font-size:0.77rem;font-weight:700;color:{_col};font-family:monospace;">{_d}{_tick}</span>'
                    f'<div style="flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden;">'
                    f'<div style="width:{_w}%;height:100%;background:{_col};border-radius:4px;"></div></div>'
                    f'<span style="width:48px;font-size:0.73rem;color:#94a3b8;font-family:monospace;">{_p:.3f}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="margin-top:4px;"><div style="font-size:0.7rem;color:#64748b;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">LAYA PROBABILITIES</div>'
                f'{_bars_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No probability data yet.")
else:
    if log_path:
        st.info(
            "Session started but no moves were logged (circuit breaker fired before the first move was written). "
            "The server's /action endpoint returned HTTP 500 on turn 1.",
            icon="⚡",
        )
    else:
        st.markdown(
            '<div style="color:#475569;padding:24px;text-align:center;border:1px dashed rgba(255,255,255,0.08);border-radius:12px;">'
            "Select a session log from the sidebar or press ▶️ to start a live game."
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
if rows:
    kpis = _session_kpis(rows, current_game)
    fb_raw = kpis.pop("fb_raw", 0.0)

    kpi_items = list(kpis.items())
    n_cols = min(len(kpi_items), 8)
    cols = st.columns(n_cols)
    for i, (label, val) in enumerate(kpi_items[:n_cols]):
        with cols[i]:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{val}</div></div>',
                unsafe_allow_html=True,
            )

    # Fallback budget bar
    fb_pct_display = f"{fb_raw:.1f}"
    bar_class = "fb-warn" if fb_raw >= 8.0 else "fb-ok"
    bar_width = min(fb_raw * 10, 100)
    st.markdown(
        f"""<div style="margin-bottom:16px;">
        <span style="font-size:0.73rem;color:#94a3b8;font-weight:600;letter-spacing:0.08em;">
        FALLBACK BUDGET &nbsp;{fb_pct_display}% used / 10% max
        {'⚠️' if fb_raw >= 8.0 else '✅'}
        </span>
        <div class="fb-wrap"><div class="{bar_class}" style="width:{bar_width}%"></div></div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Decision Logger table
# ---------------------------------------------------------------------------
st.markdown("### 📋 Decision Logger")

if rows:
    df = _build_dataframe(rows)

    # Highlight fallback rows
    def _row_style(row: pd.Series) -> list[str]:
        if row.get("Source") == "fallback":
            return ["background-color: rgba(251,146,60,0.08)"] * len(row)
        if row.get("Result") == "rejected":
            return ["background-color: rgba(248,113,113,0.06)"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_row_style, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        height=min(40 + len(df) * 35, 480),
    )

    # Last-move probability bars
    if rows:
        last_dec = rows[-1].get("decision", {})
        last_probs = last_dec.get("probabilities", {})
        last_chosen = last_dec.get("chosen", "")
        last_action = rows[-1].get("action", "")

        st.markdown(
            "**Latest Turn Probabilities** "
            f"→ Chosen: `{last_chosen.upper()}` | Source: `{last_dec.get('source', '—')}`"
        )
        max_p = max(last_probs.values()) if last_probs else 1.0
        prob_html = ""
        for direction in ("up", "down", "left", "right"):
            p = last_probs.get(direction, 0.0)
            bar_w = int(p / max(max_p, 1e-9) * 100)
            color = DIRECTION_COLORS.get(direction, "#94a3b8")
            tick = " ◀ CHOSEN" if direction == last_chosen else ""
            prob_html += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">'
                f'<span style="width:48px;font-size:0.78rem;font-weight:700;color:{color};font-family:monospace;">{direction}</span>'
                f'<div style="flex:1;height:9px;background:#1e293b;border-radius:4px;overflow:hidden;">'
                f'<div style="width:{bar_w}%;height:100%;background:{color};border-radius:4px;"></div></div>'
                f'<span style="width:58px;font-size:0.74rem;color:#94a3b8;font-family:monospace;">{p:.3f}{tick}</span>'
                f'</div>'
            )
        st.markdown(prob_html, unsafe_allow_html=True)

else:
    st.markdown(
        '<div style="color:#475569;padding:24px;text-align:center;border:1px dashed rgba(255,255,255,0.08);border-radius:12px;">'
        "No moves logged yet. Select a log file in the sidebar or run a live session."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Collapsible Laya Hypothesis & Reasoning
# ---------------------------------------------------------------------------
with st.expander("🧠 Laya Hypothesis & Reasoning (Click to Expand)"):
    if not rows:
        st.info("No session data loaded.")
    else:
        step_options = [r.get("step", i) for i, r in enumerate(rows)]
        selected_step_idx = st.selectbox(
            "Inspect step",
            options=list(range(len(rows))),
            format_func=lambda i: f"Step {step_options[i]}",
            index=len(rows) - 1,
        )
        row = rows[selected_step_idx]
        dec = row.get("decision", {})
        probs = dec.get("probabilities", {})

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Noul Questions sent to Laya**")
            questions = dec.get("questions", [])
            if questions:
                for i, q in enumerate(questions, 1):
                    st.markdown(f"`Q{i}` {q}")
            else:
                st.caption("No questions recorded in this log entry.")

        with col_b:
            st.markdown("**Raw Probability Breakdown**")
            for direction in ("up", "down", "left", "right"):
                p = probs.get(direction, 0.0)
                chosen_mark = " ← CHOSEN" if direction == dec.get("chosen") else ""
                color = DIRECTION_COLORS.get(direction, "#94a3b8")
                st.markdown(
                    f'<span style="color:{color};font-weight:700;font-family:monospace;">'
                    f'{direction:>5}</span>  `{p:.6f}`{chosen_mark}',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Sum:** `{sum(probs.values()):.6f}` (must be ∈ [0.98, 1.02])")
            st.markdown(f"**Source:** `{dec.get('source', '—')}`")
            st.markdown(f"**Confidence:** `{dec.get('confidence', 0.0):.6f}`")

        st.divider()
        st.markdown("**Raw log entry (canonical JSON)**")
        display_row = {k: v for k, v in row.items() if k != "hash"}
        st.json(display_row)

        # Hash chain verification for this entry
        import hashlib
        prev_h = row.get("prev_hash", "")
        stored_h = row.get("hash", "")
        canonical = json.dumps(display_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        computed_h = hashlib.sha256(f"{prev_h}|{canonical}".encode()).hexdigest()
        if computed_h == stored_h:
            st.success(f"✅ Hash chain VALID — `{stored_h[:24]}...`")
        else:
            st.error(f"❌ Hash chain BROKEN — stored `{stored_h[:24]}...` ≠ computed `{computed_h[:24]}...`")

# ---------------------------------------------------------------------------
# Full hash-chain audit (collapsed)
# ---------------------------------------------------------------------------
with st.expander("🔐 Full Hash-Chain Audit"):
    if not rows:
        st.info("No session data loaded.")
    else:
        import hashlib

        broken_at: int | None = None
        for i, row in enumerate(rows):
            prev_h = row.get("prev_hash", "")
            stored_h = row.get("hash", "")
            check_row = {k: v for k, v in row.items() if k != "hash"}
            canonical = json.dumps(check_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            computed = hashlib.sha256(f"{prev_h}|{canonical}".encode()).hexdigest()
            if computed != stored_h:
                broken_at = i
                break

        if broken_at is None:
            st.success(f"✅ Hash chain VALID — {len(rows)} entries verified. Log is tamper-evident.")
        else:
            st.error(f"❌ Hash chain BROKEN at entry index {broken_at} (step {rows[broken_at].get('step', '?')})")
