# 🏆 ProgrammablePros — Master Technical Briefing & Judge Knowledge Base

**Competition:** Genesis 2.0 Hackathon | **Track:** EVA 02
**Problem Statement:** *"Build the decision layer for hosted online games using the Laya calibrated probability model."*
**Team:** ProgrammablePros | **Repository:** Python 3.10+ monorepo

---

## 1. Project Overview & Zero-Mock Philosophy

The Genesis 2.0 problem statement is explicit: *"Teams using the same model will score very differently, and the games will punish weak understanding."* Our architecture is built on three non-negotiable axioms:

1. **ZERO MOCKS.** `laya.load("convaiinnovations/laya")` is called at startup. If it fails, the runner crashes loudly. There is no random number fallback, no synthetic probability generator, and no conditional branch for "mock mode" in the live runners (`run_snake_live.py`, `run_2048_live.py`).
2. **LAYA DECIDES EVERY MOVE.** Laya evaluates orthogonal `noul` (calibrated probability) hypothesis questions for every candidate action on every single turn. The EV calculator is a *weighting layer over Laya outputs* — not a replacement for them.
3. **AUDIT-GRADE LOGGING.** Every move, applied or rejected, is written to an append-only SHA-256 hash-chained `.jsonl` file before the next API call is made.

---

## 2. Live API Contract (Proven via `probe_api.py`)

**Base URL:** `https://topgameai.com/api/arena`
**Authorization:** None required. No `Authorization` header is sent.
**Content-Type:** `application/json`

### Endpoints

| Method | Path | Payload | Success Response |
|---|---|---|---|
| `POST` | `/sessions` | `{"game": "snake" or "2048", "agent_name": "ProgrammablePros"}` | `201` — `{session_id, game, state, done, won, score, steps, message}` |
| `GET` | `/sessions/{id}` | — | `200` — `{session_id, game, state, done, won, score, steps}` |
| `POST` | `/sessions/{id}/action` | `{"action": "up" or "down" or "left" or "right"}` | `200` — updated state |

### HTTP Error Contract (Discovered via Live Testing)

| Status | Meaning | Our Handling |
|---|---|---|
| `HTTP 400` | Valid rejection — 180 degree reversal in Snake, board-unchanged in 2048 | Logged as `"result": "rejected"`. Step counter not advanced. Game continues. |
| `HTTP 500` | TopGameAI server-side unhandled exception (empty body, Next.js/OpenNext/Cloudflare edge crash). Our payload is valid. | **Circuit breaker fires instantly.** Prints `[!] FATAL: Session corrupted on server (HTTP 500). Aborting.` Breaks the game loop. `main()` auto-restarts a fresh session (up to 3 attempts). |
| `HTTP 404` | Session not found | Raised as `GameAPIError`. Propagates to `main()`. |

> **Proof:** `probe_api.py` creates a fresh session and tests every action payload variant against the live endpoint. It confirmed: `{"action": "left"}` produced HTTP 500 (empty body) during the Oct 17 arena outage, while `POST /sessions` continued to return HTTP 201. The `/action` endpoint was broken server-side. Our circuit breaker detected this in under 1 second instead of wasting 30+ seconds on futile retries.

### Why urllib3 Auto-Retry Was Disabled for HTTP 500

`game_clients._build_retry_session()` uses `status_forcelist=[429, 502, 503, 504]`. HTTP 500 is intentionally excluded. If 500 were in the list, urllib3 would silently retry 3 times (spending ~15 seconds) before our application-level circuit breaker could fire. Removing 500 from the forcelist means a 500 surfaces immediately to our runner's `except GameAPIError` block.

---

## 3. Architecture — Full Decision Pipeline

```
 +==========================================================================+
 |         PROGRAMMABLEPROS -- LAYA EVA 02 DECISION PIPELINE                |
 +==========================================================================+

  TopGameAI Arena API  -->  Raw JSON Game State
                                     |
                                     v
 +------------------------------------------------------------------------+
 |  STAGE 1 -- SENSE & COMPRESS                                           |
 |  SnakeCompressor / Board2048Compressor                                 |
 |  Strips nested UI metadata -> <=200-char high-signal context string    |
 |  Snake:  "H:(6,6) D:right L:3 F:(10,0) dF:10 W:6/13/6/6 E:0 S:297"  |
 |  2048:   "T:64(2^6)@BL E:11 Sm:2.1 M:3.4 Sc:384 S:1996"             |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 2 -- REASON & HYPOTHESIZE                                       |
 |  SnakeSubgoalManager / Board2048SubgoalManager                         |
 |  -> Active subgoal: CHASE_FOOD | SURVIVE | KEEP_CORNER                 |
 |  SnakeHypothesisBuilder / Board2048HypothesisBuilder                   |
 |  -> 8 typed noul questions (2 per action: Q1=progress, Q2=danger)      |
 |    Q1: "Moving '{a}' is legal and progresses toward {subgoal}."        |
 |    Q2: "Moving '{a}' causes immediate collision or board paralysis."   |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 3 -- REAL LAYA INFERENCE                                        |
 |  laya.load("convaiinnovations/laya")  <--  Hugging Face local cache    |
 |  LayaAgent.predict(compressed_state, questions)                        |
 |  -> {"q_up_progress": 0.74, "q_up_danger": 0.12, ...} (~1300ms/turn) |
 |  ZERO MOCKS: RuntimeError raised if laya package is not installed      |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 4 -- MULTI-TERM EV DECISION                                     |
 |  SnakeEVCalculator / Board2048EVCalculator                             |
 |  EV(a) = P_progress(a)*W_prog + P_struct(a)*W_struct                  |
 |           - P_danger(a)*W_danger - penalty_repeat(a)                  |
 |  -> proposed_action = argmax EV over {up, down, left, right}           |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 5 -- SAFETY VALIDATION (<10% FALLBACK BUDGET)                  |
 |  Snake: Anti-reversal interceptor (180 degree = instant kill, HTTP 400)|
 |  2048:  Anti-no-op interceptor (unchanged board = HTTP 400)            |
 |  Action sanitizer: strip/lowercase, reject anything outside legal set  |
 |  Pre-move GET /sessions/{id} check: abort if done=True before sending  |
 |  If fallback used: source="fallback", logged explicitly                |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 6 -- COMPLIANCE & NORMALIZATION                                 |
 |  ComplianceLayer.normalize_laya_probabilities()                        |
 |  -> 4-way sum enforced in [0.98, 1.02], residual absorbed into max    |
 |  ComplianceLayer.build_decision_block()                                |
 |  -> {questions[], probabilities{up,down,left,right}, chosen,           |
 |      confidence, source}                                               |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 7 -- DISPATCH & CIRCUIT BREAKER                                 |
 |  GET /sessions/{id}  -> verify session alive (done != True)            |
 |  POST /sessions/{id}/action  ->  {"action": final_action}             |
 |  HTTP 400 -> "rejected", loop continues                                |
 |  HTTP 500 -> circuit breaker fires, break, main() opens fresh session  |
 +-------------------------------------+----------------------------------+
                                       |
                                       v
 +------------------------------------------------------------------------+
 |  STAGE 8 -- SHA-256 HASH-CHAIN LOGGING (every move, applied+rejected) |
 |  H_0 = "GENESIS-" + session_id                                         |
 |  H_n = SHA256(H_{n-1} + "|" + canonical_json(line_minus_hash_field))  |
 |  Fields: session_id, game, step, state_before, action, state_after,   |
 |           result, decision, timing, ts (ISO 8601 UTC), prev_hash, hash |
 |  -> logs/{game}/decision_log_{session_id}.jsonl                        |
 +------------------------------------------------------------------------+
```

---

## 4. Proof of Real Laya Implementation

### 4.1 Model Loading — laya_agent.py Lines 38–44, 64, 92

```python
# Hard import -- RuntimeError if 'laya' package not installed
import laya  # raises ImportError -> RuntimeError on failure

class LayaAgent:
    MODEL_ID = "convaiinnovations/laya"

    def __init__(self, model_id=MODEL_ID, ...):
        self.agent = laya.load(self.model_id, device=self.device)
        # Fails loudly if weights missing from Hugging Face local cache
```

There is no conditional path, no random fallback, and no `MockLayaBackend` in `run_snake_live.py` or `run_2048_live.py`. `MockLayaBackend` in `core/laya_backend.py` raises `RuntimeError` on `__init__` and exists only in `tests/`.

### 4.2 Per-Turn Terminal Proof

Every turn prints:

```
=== REAL LAYA INFERENCE EXECUTING ===
=== REAL LAYA INFERENCE COMPLETE (1338ms) ===
```

Latency of ~1300ms is the expected hardware inference time for `convaiinnovations/laya` on CPU evaluating 8 noul questions per turn.

### 4.3 probe_api.py — Live API Diagnostic Tool

Located at `tools/probe_api.py`. Creates a real session and fires every known payload variant at the `/action` endpoint, logging exact HTTP status codes and response bodies. Used on Oct 17 2026 to confirm a server-side outage on TopGameAI's `/action` endpoint while `POST /sessions` continued to return HTTP 201.

---

## 5. Logging Specification (Genesis 2.0 Compliance)

Each `.jsonl` line is one complete JSON object. Example:

```json
{
  "session_id": "62fa035a-9b9c-4213-be70-52e335b952aa",
  "game": "snake",
  "step": 14,
  "ts": "2026-10-17T04:38:39.123Z",
  "state_before": {},
  "action": "right",
  "state_after": {},
  "result": "applied",
  "decision": {
    "questions": [
      "Moving 'up' is legal and progresses toward food at (10, 0).",
      "Moving 'up' causes immediate collision or board paralysis."
    ],
    "probabilities": {"up": 0.2432, "down": 0.2541, "left": 0.2497, "right": 0.2530},
    "chosen": "right",
    "confidence": 0.2530,
    "source": "laya"
  },
  "timing": {"decide_ms": 1338, "api_ms": 212, "turn_ms": 1552},
  "prev_hash": "GENESIS-62fa035a-...",
  "hash": "a3f9c2d1..."
}
```

**Invariants enforced at write time by `OfficialLogger`:**

- `chosen == action`
- `confidence == probabilities[chosen]`
- `sum(probabilities.values())` is in `[0.98, 1.02]`
- `source` is one of `"laya"` or `"fallback"`
- `fallback_count / total_moves <= 0.10`
- `hash = SHA256(prev_hash + "|" + canonical_json_without_hash_field)`

---

## 6. File Inventory

| File | Key Class / Entry Point | Role |
|---|---|---|
| `run_snake_live.py` | `run_live_snake_session()`, `main()` | Stage 1 live runner. Laya + SnakeClient + SHA-256 logging. |
| `run_2048_live.py` | `run_live_2048_session()`, `main()` | Stage 2 live runner. Laya + Game2048Client + SHA-256 logging. |
| `laya_agent.py` | `LayaAgent` | Loads `convaiinnovations/laya`, formats noul questions, runs inference, tracks failure memory. |
| `game_clients.py` | `SnakeClient`, `Game2048Client`, `BaseGameClient` | HTTP client for TopGameAI Arena. Circuit breaker, 400/500 handling, exponential backoff on 429/502-504. |
| `snake_brain.py` | `SnakeCompressor`, `SnakeSubgoalManager`, `SnakeHypothesisBuilder`, `SnakeEVCalculator` | Full Stage 1 perception to reasoning to EV pipeline. |
| `game2048_brain.py` | `Board2048Compressor`, `Board2048HypothesisBuilder`, `Board2048EVCalculator` | Full Stage 2 perception to reasoning to EV pipeline. |
| `compliance_layer.py` | `ComplianceLayer` | 4-way probability normalization, fallback guard, canonical decision block builder. |
| `official_logger.py` | `OfficialLogger` | SHA-256 hash-chained append-only JSONL logger. Includes `verify_log_file()`. |
| `viewer_local.py` | Streamlit app | Live telemetry dashboard — boards, probability bars, EV telemetry, audit chain feed. |
| `run_local_games.py` | `run_session()` | Local test harness runner using real Laya model and local game engine. |
| `tools/probe_api.py` | standalone script | Live API endpoint diagnostic tool. Proves server contract and detects outages. |

---

## 7. Running the Live Agents

### Stage 1 — Snake

```bash
python run_snake_live.py
python run_snake_live.py --agent-name ProgrammablePros --max-steps 300 --log-dir logs/snake --max-restarts 3
```

### Stage 2 — 2048

```bash
python run_2048_live.py
python run_2048_live.py --agent-name ProgrammablePros --max-steps 2000 --log-dir logs/2048
```

### Live Telemetry Viewer

```bash
streamlit run app.py
# Open: http://localhost:8501
```

### Local Benchmark (Real Laya, Local Game Engine)

```bash
python run_local_games.py --game snake --seed 42
python run_local_games.py --game 2048 --seed 42
```

---

## 8. Game Rules Reference

| | Stage 1: Snake | Stage 2: 2048 |
|---|---|---|
| Grid | 20x20 | 4x4 |
| Win condition | 15 foods eaten | Reach 2048 tile |
| Move cap | 300 accepted steps | 2000 accepted steps |
| Illegal moves | 180 degree reversal = HTTP 400 | Board unchanged = HTTP 400 |
| Scoring | foods x 60 (+100 if win), capped 1000 | min(1000, max(0, (log2(best_tile) - 6) x 143 + floor(score/1000))) |
| Stage unlock | 600+ points (10+ foods) | Reach 512 tile |

---

## 9. Judge Q&A

**Q: How do we know you are not using a mock model?**

`laya_agent.py` line 38: `import laya` — hard import, crashes if not installed. Line 92: `self.agent = laya.load(self.model_id, ...)` — hard model load, crashes if weights missing. The terminal prints `LOADING REAL LAYA MODEL: convaiinnovations/laya` with measured load time (~4700ms on CPU). There is no path in `run_snake_live.py` or `run_2048_live.py` that bypasses this.

**Q: Why does Laya take ~1300ms per turn?**

`convaiinnovations/laya` runs on CPU in the hackathon environment. Each turn evaluates 8 noul questions (2 per action x 4 actions) in a single forward pass. The arena has no per-move time limit.

**Q: What happens when TopGameAI returns HTTP 500?**

The circuit breaker in `run_snake_live.py` and `run_2048_live.py` catches `GameAPIError` with `status_code >= 500` inside the game loop and immediately breaks — no retries, no sleep, no exception propagation. The session log is closed cleanly, and `main()` automatically opens a fresh session (up to `--max-restarts` attempts, default 3).

**Q: How is the fallback rate guaranteed to stay under 10%?**

`ComplianceLayer` tracks `fallback_count / total_moves` and raises `ComplianceError` if it would exceed 10%. In practice, the EV calculator's legality mask prevents illegal moves before they reach the API, keeping the fallback rate at 0% in benchmarks.

**Q: Why is EV calculation better than argmax on Laya probabilities?**

An argmax bot blindly picks the highest-probability move. If that move carries a catastrophic wall-crash danger penalty, its true expected value is deeply negative. A slightly lower-probability move with no danger yields positive EV. Our EV engine calculates mathematical expectation over both progress and danger signals simultaneously, eliminating high-confidence suicide moves that naive argmax would select.
