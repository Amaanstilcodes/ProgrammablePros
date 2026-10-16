# TopGameAI Arena API Contract & Ambiguity Resolution

**Genesis 2.0 — Track EVA 02**  
**Team:** ProgrammablePros  
**Target Games:** Stage 1 (Snake) & Stage 2 (2048)  
**Host URL:** `https://topgameai.com/api`

---

## 1. Endpoints

### 1.1 Stage 1: Snake
- `POST /snake/start`
  - Request: `{"agent": "ProgrammablePros", "game": "snake"}`
  - Response: `{"session_id": "<sid>", "state": {...}}`
- `POST /snake/move`
  - Request: `{"session_id": "<sid>", "action": "up"|"down"|"left"|"right"}`
  - Response: `{"result": "applied"|"rejected", "state": {...}, "game_over": bool}`
- `GET /snake/state?session_id=<sid>`
  - Response: `{"state": {...}}`

### 1.2 Stage 2: 2048
- `POST /2048/start`
  - Request: `{"agent": "ProgrammablePros", "game": "2048"}`
  - Response: `{"session_id": "<sid>", "state": {...}}`
- `POST /2048/move`
  - Request: `{"session_id": "<sid>", "action": "up"|"down"|"left"|"right"}`
  - Response: `{"result": "applied"|"rejected", "state": {...}, "game_over": bool}`

---

## 2. Step Accounting & Rejection Policies

### Ambiguity Flagged to Organizers
The PDF rulebook states:
> *"One line per APPLIED move, AND one line per REJECTED move... step starts at 1 with no gaps across applied sequence... max 300 accepted moves for Snake."*

If a move is rejected, how is the integer `step` field numbered? We implement a configurable `StepLedger` with two standard policies:

### Policy A (Default): `"repeat_last_applied"`
- `next_step` starts at `1`.
- On `applied`: `step = next_step; next_step += 1`.
- On `rejected`: `step = max(1, next_step - 1); next_step` does NOT increment.
- **Result:** The applied moves form a strictly monotonic `1..N` sequence with zero gaps. The final log line's `step` equals the accepted move count.

### Policy B (Alternate): `"pending_ordinal"`
- On `rejected`: `step = next_step; next_step` does NOT increment.
- **Configurable via:** `REJECTED_STEP_POLICY="repeat_last_applied"` or `"pending_ordinal"`.

---

## 3. Strict Compliance Invariants

| ID | Invariant | Enforcement Mechanism |
|---|---|---|
| **I1** | Exactly 4 keys: `up`, `down`, `left`, `right` | `ComplianceLayer.normalize_probabilities()` |
| **I2** | Finite probabilities $\in [0.0, 1.0]$ | Clamped with NaN/Inf guards |
| **I3** | Sum strictly $\in [0.98, 1.02]$ | Residual absorbed into dominant action to guarantee $1.000000$ |
| **I4** | `decision.chosen == log_line.action` | Exact string equality verified |
| **I5** | `decision.confidence == probabilities[chosen]` | Exact float lookup verified |
| **I6** | `decision.source in {"laya", "fallback"}` | Strict enum validation |
| **I7** | Fallback moves $\le 10\%$ of accepted moves | Tracked per session |
| **I8** | Zero `NaN` / `Infinity` in JSON | Canonical JSON sanitizer raises `ValueError` |
| **I9** | Unmodified deep copies of raw API state | `copy.deepcopy(raw_state)` |
| **I10** | Cryptographic hash continuity | `hash = SHA256(prev_hash + "\|" + canonical_json(line))` |
