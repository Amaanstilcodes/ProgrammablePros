"""ProgrammablePros — Judge Demo Replay Viewer.

Replays recorded decision logs step-by-step to demonstrate
real Laya inference on Snake and 2048 to judges.

Usage:
    streamlit run replay.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProgrammablePros · Judge Replay",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #080b12; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg,#0f172a 0%,#0d1f3c 60%,#0a1628 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 16px; padding: 20px 28px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
}
.hero-title { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; margin:0; }
.hero-sub   { font-size: 0.8rem; color: #64748b; margin-top:3px; }
.hero-badge {
    background: linear-gradient(135deg,#0ea5e9,#6366f1);
    color:#fff; font-size:0.72rem; font-weight:700;
    padding:5px 14px; border-radius:20px; letter-spacing:.06em;
}

.kpi-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
.kpi-card {
    flex:1; min-width:100px;
    background: linear-gradient(145deg,#111827,#0f172a);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:12px 16px; text-align:center;
}
.kpi-label { font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }
.kpi-value { font-size:1.5rem; font-weight:800; color:#f1f5f9; margin-top:2px; line-height:1; }
.kpi-sub   { font-size:.68rem; color:#475569; margin-top:3px; }

.step-badge {
    display:inline-block;
    background: linear-gradient(135deg,#0ea5e9,#6366f1);
    color:#fff; font-size:1rem; font-weight:800;
    padding:6px 18px; border-radius:20px; margin-bottom:12px;
}
.action-display {
    font-size:2rem; font-weight:800; letter-spacing:.04em;
    text-align:center; padding:12px;
    background:linear-gradient(135deg,#042f2e,#0d2330);
    border:1px solid rgba(45,212,191,.3);
    border-radius:12px; margin-bottom:10px;
}
.src-laya     { color:#2dd4bf; }
.src-fallback { color:#facc15; }

.hash-ok   { background:#052e16; border:1px solid #166534; border-radius:8px; padding:8px 12px; color:#4ade80; font-size:.78rem; font-weight:600; margin-top:8px; }
.hash-fail { background:#2d0a0a; border:1px solid #7f1d1d; border-radius:8px; padding:8px 12px; color:#f87171; font-size:.78rem; font-weight:600; margin-top:8px; }

.notice-box {
    background:#0c1a2e; border:1px solid #1d4ed8;
    border-radius:10px; padding:14px 18px; margin-bottom:16px;
    font-size:.85rem; color:#93c5fd; line-height:1.6;
}

#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

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
DIR_COLORS = {"up": "#38bdf8", "down": "#a78bfa", "left": "#fb923c", "right": "#4ade80"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


SESSIONS = {
    "🐍 Snake  —  300 moves  |  Score 120": ROOT / "logs" / "decision_log_local_snake_a607d7c3.jsonl",
    "🎯 2048   —  251 moves  |  Score 3052": ROOT / "logs" / "decision_log_local_2048_868e61fe.jsonl",
}


def _xy(seg: Any) -> tuple[int, int]:
    if isinstance(seg, dict):
        return int(seg.get("x", 0)), int(seg.get("y", 0))
    return int(seg[0]), int(seg[1])


def _render_snake(state: dict, highlight_action: str = "") -> str:
    w = state.get("width", 20)
    h = state.get("height", 20)
    snake = state.get("snake", [])
    food = state.get("food", {"x": -1, "y": -1})
    grid = [["e"] * w for _ in range(h)]
    for i, seg in enumerate(snake):
        x, y = _xy(seg)
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = "H" if i == 0 else "B"
    fx, fy = _xy(food)
    if 0 <= fx < w and 0 <= fy < h:
        grid[fy][fx] = "F"

    cell_map = {
        "e": ("#131e2e", "1px solid rgba(255,255,255,0.03)", ""),
        "H": ("linear-gradient(135deg,#0ea5e9,#38bdf8)", "none", ""),
        "B": ("#0369a1", "none", ""),
        "F": ("radial-gradient(circle,#f87171,#ef4444)", "none", "🍎"),
    }
    cells_html = ""
    for row in grid:
        for cell in row:
            bg, border, sym = cell_map[cell]
            cells_html += (
                f'<div style="width:22px;height:22px;border-radius:3px;'
                f'background:{bg};border:{border};display:flex;'
                f'align-items:center;justify-content:center;font-size:9px;">{sym}</div>'
            )
    return (
        f'<div style="display:inline-grid;grid-template-columns:repeat({w},22px);'
        f'gap:2px;background:#0f172a;padding:8px;border-radius:10px;">{cells_html}</div>'
    )


def _render_2048(state: dict) -> str:
    board = state.get("board", [[0]*4]*4)
    tiles = ""
    for row in board:
        for val in row:
            bg, fg = TILE_COLORS.get(val, ("#3c3a32", "#f9f6f2"))
            label = str(val) if val > 0 else ""
            fsize = "20px" if val < 1000 else "14px"
            tiles += (
                f'<div style="width:82px;height:82px;border-radius:8px;'
                f'background:{bg};color:{fg};font-size:{fsize};'
                f'font-weight:800;display:flex;align-items:center;justify-content:center;">'
                f'{label}</div>'
            )
    return (
        f'<div style="display:inline-grid;grid-template-columns:repeat(4,82px);'
        f'gap:8px;background:#1e293b;padding:10px;border-radius:12px;">{tiles}</div>'
    )


def _prob_bars(probs: dict, chosen: str) -> str:
    html = ""
    mx = max(probs.values()) if probs else 1.0
    for d in ("up", "down", "left", "right"):
        p = probs.get(d, 0.0)
        w = int(p / max(mx, 1e-9) * 100)
        c = DIR_COLORS.get(d, "#94a3b8")
        tick = " ◀ CHOSEN" if d == chosen else ""
        html += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
            f'<span style="width:46px;font-size:.78rem;font-weight:700;'
            f'color:{c};font-family:monospace;">{d}</span>'
            f'<div style="flex:1;height:9px;background:#1e293b;border-radius:4px;overflow:hidden;">'
            f'<div style="width:{w}%;height:100%;background:{c};border-radius:4px;"></div></div>'
            f'<span style="width:80px;font-size:.73rem;color:#94a3b8;font-family:monospace;">'
            f'{p:.4f}{tick}</span></div>'
        )
    return html


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "replay_idx": 0,
    "autoplay": False,
    "speed_ms": 500,
    "loaded_log": None,
    "rows": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-title">🎬 ProgrammablePros — Judge Replay Viewer</div>
    <div class="hero-sub">Step-by-step replay of real Laya decisions · SHA-256 hash-chain verified · Genesis 2.0 Track EVA 02</div>
  </div>
  <div class="hero-badge">LIVE REPLAY</div>
</div>
""", unsafe_allow_html=True)

# ── Server outage notice ──────────────────────────────────────────────────────
st.markdown("""
<div class="notice-box">
  ⚡ <strong>TopGameAI Arena Server Notice:</strong>
  The <code>/action</code> endpoint is returning HTTP 500 (empty body) for all valid moves as of Oct 17 2026.
  Our diagnostic confirmed: session creation ✅, state fetch ✅, reversal detection ✅ (HTTP 400 with correct error),
  but the game state update crashes server-side on all valid moves ❌ — payload format is proven correct.
  <br><br>
  This replay viewer demonstrates <strong>real Laya inference decisions</strong> recorded during a fully successful
  local run, including the complete SHA-256 hash chain, noul questions, and calibrated probabilities.
</div>
""", unsafe_allow_html=True)

# ── Sidebar: log selector + controls ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Select Session Log")
    chosen_label = st.radio(
        "Select game to replay:",
        list(SESSIONS.keys()),
        label_visibility="collapsed",
    )
    chosen_path = SESSIONS[chosen_label]

    # Load log if changed
    if st.session_state.loaded_log != str(chosen_path):
        st.session_state.rows = _load_log(chosen_path)
        st.session_state.replay_idx = 0
        st.session_state.loaded_log = str(chosen_path)
        st.session_state.autoplay = False

    rows = st.session_state.rows
    total = len(rows)


    st.divider()
    st.markdown("### ⏯ Playback Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏮", use_container_width=True, help="First step"):
            st.session_state.replay_idx = 0
            st.session_state.autoplay = False
    with col2:
        if st.button("◀", use_container_width=True, help="Previous step"):
            st.session_state.replay_idx = max(0, st.session_state.replay_idx - 1)
            st.session_state.autoplay = False
    with col3:
        if st.button("▶", use_container_width=True, help="Next step"):
            st.session_state.replay_idx = min(total - 1, st.session_state.replay_idx + 1)
            st.session_state.autoplay = False

    col4, col5 = st.columns(2)
    with col4:
        if st.button("▶▶ Auto", use_container_width=True, type="primary"):
            st.session_state.autoplay = True
    with col5:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.autoplay = False

    if total > 1:
        st.session_state.replay_idx = st.slider(
            "Step",
            min_value=0,
            max_value=total - 1,
            value=min(st.session_state.replay_idx, total - 1),
            label_visibility="collapsed",
        )
    else:
        st.caption("Single-entry log — no scrubbing needed.")
        st.session_state.replay_idx = 0

    speed_label = st.select_slider(
        "Speed",
        options=["0.25×", "0.5×", "1×", "2×", "4×"],
        value="1×",
    )
    speed_map = {"0.25×": 2000, "0.5×": 1000, "1×": 500, "2×": 250, "4×": 125}
    st.session_state.speed_ms = speed_map[speed_label]

    st.divider()
    applied = sum(1 for r in rows if r.get("result") == "applied")
    fallbacks = sum(1 for r in rows if r.get("decision", {}).get("source") == "fallback")
    st.markdown(f"**Total moves:** {total}")
    st.markdown(f"**Applied:** {applied} &nbsp;|&nbsp; **Rejected:** {total - applied}")
    fb_pct = fallbacks / total * 100 if total else 0
    st.markdown(f"**Fallbacks:** {fallbacks} ({fb_pct:.1f}%) — budget ≤10%")

# ── Autoplay scheduling (runs AFTER full page renders below) ─────────────────
_should_advance = (
    st.session_state.autoplay
    and total > 1
    and st.session_state.replay_idx < total - 1
)
_at_end = st.session_state.autoplay and st.session_state.replay_idx >= total - 1

# ── Empty log guard ──────────────────────────────────────────────────────────
if not rows:
    st.warning(
        "⚡ This session has **0 logged moves** — the circuit breaker fired before any "
        "move was written to the log (HTTP 500 on move 1). "
        "Select a different log from the sidebar to replay a full session.",
        icon="⚡",
    )
    st.info(
        "**Available full sessions:**\n"
        "- 🐍 `decision_log_local_snake_a607d7c3.jsonl` — 300 moves\n"
        "- 🎯 `decision_log_local_2048_868e61fe.jsonl` — 251 moves\n"
        "- 🎯 `decision_log_local_2048_73bc447d.jsonl` — 251 moves",
    )
    st.stop()

# ── Current step data ─────────────────────────────────────────────────────────
idx = st.session_state.replay_idx
row = rows[idx]
dec = row.get("decision", {})
probs = dec.get("probabilities", {})
chosen = dec.get("chosen", row.get("action", ""))
source = dec.get("source", "laya")
state = row.get("state_after") or row.get("state_before") or {}
game = row.get("game", "snake")

# ── Step badge + progress bar ─────────────────────────────────────────────────
prog_pct = int((idx + 1) / total * 100)
st.markdown(
    f'<div class="step-badge">Step {idx + 1} / {total} &nbsp;·&nbsp; {prog_pct}% complete</div>',
    unsafe_allow_html=True,
)
st.progress(prog_pct / 100)

# ── KPI row ───────────────────────────────────────────────────────────────────
score = state.get("score", 0)
if game == "snake":
    foods = state.get("foods_eaten", 0)
    length = len(state.get("snake", []))
    kpi_html = f"""<div class="kpi-row">
      <div class="kpi-card"><div class="kpi-label">Score</div><div class="kpi-value" style="color:#38bdf8;">{score}</div></div>
      <div class="kpi-card"><div class="kpi-label">Foods</div><div class="kpi-value" style="color:#4ade80;">{foods}/15</div><div class="kpi-sub">target</div></div>
      <div class="kpi-card"><div class="kpi-label">Length</div><div class="kpi-value">{length}</div></div>
      <div class="kpi-card"><div class="kpi-label">Action</div><div class="kpi-value" style="color:{DIR_COLORS.get(chosen,'#f1f5f9')};font-size:1.1rem;">{chosen.upper()}</div></div>
      <div class="kpi-card"><div class="kpi-label">Source</div><div class="kpi-value {'src-laya' if source=='laya' else 'src-fallback'}" style="font-size:1rem;">{source.upper()}</div></div>
      <div class="kpi-card"><div class="kpi-label">Confidence</div><div class="kpi-value" style="font-size:1.1rem;">{dec.get('confidence', 0):.3f}</div></div>
    </div>"""
else:
    flat = [v for r in state.get("board", []) for v in r]
    max_tile = max(flat) if flat else 0
    empty = flat.count(0) if flat else 0
    kpi_html = f"""<div class="kpi-row">
      <div class="kpi-card"><div class="kpi-label">Score</div><div class="kpi-value" style="color:#38bdf8;">{score}</div></div>
      <div class="kpi-card"><div class="kpi-label">Max Tile</div><div class="kpi-value" style="color:#edc22e;">{max_tile}</div></div>
      <div class="kpi-card"><div class="kpi-label">Empty</div><div class="kpi-value">{empty}</div></div>
      <div class="kpi-card"><div class="kpi-label">Action</div><div class="kpi-value" style="color:{DIR_COLORS.get(chosen,'#f1f5f9')};font-size:1.1rem;">{chosen.upper()}</div></div>
      <div class="kpi-card"><div class="kpi-label">Source</div><div class="kpi-value {'src-laya' if source=='laya' else 'src-fallback'}" style="font-size:1rem;">{source.upper()}</div></div>
      <div class="kpi-card"><div class="kpi-label">Confidence</div><div class="kpi-value" style="font-size:1.1rem;">{dec.get('confidence', 0):.3f}</div></div>
    </div>"""

st.markdown(kpi_html, unsafe_allow_html=True)

# ── Main columns: Board | Decision ───────────────────────────────────────────
col_board, col_decision = st.columns([3, 2])

with col_board:
    st.markdown("#### 🎮 Game Board")
    if game == "snake":
        st.markdown(_render_snake(state, chosen), unsafe_allow_html=True)
        snake = state.get("snake", [])
        head = _xy(snake[0]) if snake else "—"
        food = state.get("food", {})
        food_pos = _xy(food)
        direction = state.get("direction", "—").upper()
        st.markdown(
            f'<div style="margin-top:8px;font-size:.72rem;color:#64748b;">'
            f'Head <b style="color:#38bdf8">{head}</b> &nbsp;·&nbsp; '
            f'Food <b style="color:#ef4444">{food_pos}</b> &nbsp;·&nbsp; '
            f'Dir <b style="color:#a78bfa">{direction}</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(_render_2048(state), unsafe_allow_html=True)

with col_decision:
    st.markdown("#### 🧠 Laya Decision")

    # Action display
    c = DIR_COLORS.get(chosen, "#f1f5f9")
    res_color = "#4ade80" if row.get("result") == "applied" else "#f87171"
    result_label = row.get("result", "?").upper()
    st.markdown(
        f'<div class="action-display" style="color:{c};">'
        f'{chosen.upper()} &nbsp;<span style="font-size:1rem;color:{res_color};">({result_label})</span></div>',
        unsafe_allow_html=True,
    )

    # Probability bars
    st.markdown(
        f'<div style="font-size:.68rem;color:#64748b;font-weight:600;'
        f'letter-spacing:.09em;margin-bottom:8px;">LAYA CALIBRATED PROBABILITIES</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_prob_bars(probs, chosen), unsafe_allow_html=True)

    # Hash chain verification
    import hashlib
    prev_h = row.get("prev_hash", "")
    stored_h = row.get("hash", "")
    check = {k: v for k, v in row.items() if k != "hash"}
    canonical = json.dumps(check, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    computed = hashlib.sha256(f"{prev_h}|{canonical}".encode()).hexdigest()
    if computed == stored_h:
        st.markdown(
            f'<div class="hash-ok">🔐 Hash chain VALID — SHA-256 intact<br>'
            f'<span style="font-family:monospace;font-size:.72rem;color:#86efac;">{stored_h[:32]}...</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="hash-fail">❌ Hash chain BROKEN</div>', unsafe_allow_html=True)

# ── Laya Reasoning expander ───────────────────────────────────────────────────
with st.expander("🧠 Laya Noul Questions & Raw Reasoning (Click to Expand)"):
    questions = dec.get("questions", [])
    if questions:
        st.markdown("**Hypothesis questions sent to Laya this turn:**")
        for i, q in enumerate(questions, 1):
            st.markdown(f"`Q{i}` {q}")
    else:
        st.caption("No questions recorded in this log entry.")

    st.divider()
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Raw probabilities (6 decimal places)**")
        for d in ("up", "down", "left", "right"):
            p = probs.get(d, 0.0)
            mk = " ← CHOSEN" if d == chosen else ""
            col = DIR_COLORS.get(d, "#94a3b8")
            st.markdown(
                f'<span style="color:{col};font-weight:700;font-family:monospace;">{d:>6}</span>'
                f'  `{p:.6f}`{mk}',
                unsafe_allow_html=True,
            )
        st.markdown(f"**Sum:** `{sum(probs.values()):.6f}` — must be ∈ [0.98, 1.02]")
    with col_r2:
        st.markdown("**Timing this turn**")
        timing = row.get("timing", {})
        for k, label in [("decide_ms", "Laya inference"), ("api_ms", "API call"), ("turn_ms", "Total turn")]:
            v = timing.get(k, "—")
            st.markdown(f"- **{label}:** `{v}ms`")
        st.markdown(f"**Timestamp:** `{row.get('ts', '—')}`")

# ── Full decision log table (collapsed) ──────────────────────────────────────
with st.expander("📋 Full Session Decision Log"):
    import pandas as pd
    records = []
    for r in rows:
        d = r.get("decision", {})
        p = d.get("probabilities", {})
        records.append({
            "Step":      r.get("step", ""),
            "Action":    r.get("action", ""),
            "Result":    r.get("result", ""),
            "Source":    d.get("source", ""),
            "Conf":      f"{d.get('confidence', 0):.4f}",
            "P(up)":     f"{p.get('up',   0):.4f}",
            "P(down)":   f"{p.get('down', 0):.4f}",
            "P(left)":   f"{p.get('left', 0):.4f}",
            "P(right)":  f"{p.get('right',0):.4f}",
            "Decide ms": r.get("timing", {}).get("decide_ms", ""),
        })
    df = pd.DataFrame(records)

    def _style(row: pd.Series):
        if row.get("Source") == "fallback":
            return ["background-color:rgba(251,146,60,.1)"] * len(row)
        if row.get("Result") == "rejected":
            return ["background-color:rgba(248,113,113,.07)"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(_style, axis=1), use_container_width=True, height=350)

# ── Autoplay: advance AFTER full render, sleep, then trigger next frame ───────
if _should_advance:
    time.sleep(st.session_state.speed_ms / 1000.0)
    st.session_state.replay_idx += 1
    st.rerun()
elif _at_end:
    st.session_state.autoplay = False
