# TopGameAI Arena API Contract & Ambiguity Resolution

**Genesis 2.0 — Track EVA 02**  
**Team:** ProgrammablePros  
**Target Games:** Stage 1 (Snake) & Stage 2 (2048)  
**Host URL:** `https://topgameai.com/api` (Configurable via `TOPGAMEAI_BASE_URL`)

---

## 1. Base URL & Authentication Mechanism

- **Base URL:** Defined via environment variable `TOPGAMEAI_BASE_URL` (default: `https://topgameai.com`).
- **Auth Header:** `Authorization: Bearer <TOPGAMEAI_API_KEY>` (or `x-api-key: <KEY>` depending on live probe).
- **Format:** HTTP Bearer token.
- **Redaction Invariant:** Authentication tokens must NEVER be logged to plaintext JSONL or rich traces.

---

## 2. Endpoint Table

*Note: All paths below are provisional until verified by `tools/probe_api.py`.*

| Purpose | Method | Path | Request Schema | Response Schema | Status |
|---|---|---|---|---|---|
| Start Snake Session | `POST` | `/api/arena/snake/start` | `{"agent": "ProgrammablePros"}` | `{"session_id": str, "state": dict}` | UNVERIFIED — capture required |
| Send Snake Move | `POST` | `/api/arena/snake/move` | `{"session_id": str, "action": str}` | `{"result": "applied"\|"rejected", "state": dict, "game_over": bool}` | UNVERIFIED — capture required |
| Inspect Snake State | `GET` | `/api/arena/snake/state` | Query param: `?session_id=<sid>` | `{"state": dict}` | UNVERIFIED — capture required |
| Start 2048 Session | `POST` | `/api/arena/2048/start` | `{"agent": "ProgrammablePros"}` | `{"session_id": str, "state": dict}` | UNVERIFIED — capture required |
| Send 2048 Move | `POST` | `/api/arena/2048/move` | `{"session_id": str, "action": str}` | `{"result": "applied"\|"rejected", "state": dict, "game_over": bool}` | UNVERIFIED — capture required |
| Inspect 2048 State | `GET` | `/api/arena/2048/state` | Query param: `?session_id=<sid>` | `{"state": dict}` | UNVERIFIED — capture required |

---

## 3. Session Lifecycle

- **Creation:** Initiated via POST to `/start` with team credentials (`ProgrammablePros`). Server allocates a unique `session_id`.
- **Addressing:** Every subsequent move and polling request must pass `session_id` (either in JSON body or query param).
- **Termination:** Triggered when `game_over == True`, maximum step cap is hit (Snake: 300 accepted moves; 2048: 2000 accepted moves), or client issues session closure.
- **Persistence:** In-memory on arena server during match; final state persisted via `final_state_{session_id}.json`.

---

## 4. Snake State Schema

| Field / Aspect | Specification / Questions | Verified Status |
|---|---|---|
| **Body representation** | List of coordinates `[[x, y], ...]` | UNVERIFIED — capture required |
| **Head element** | Index 0 assumed as head (`snake[0]`) | UNVERIFIED — capture required |
| **Direction** | Lowercase string: `"up"`, `"down"`, `"left"`, `"right"` | UNVERIFIED — capture required |
| **Food position** | Coordinate pair `[x, y]` | UNVERIFIED — capture required (Parser supports `food=None` fallback) |
| **Score key** | Integer under `"score"` | UNVERIFIED — capture required |
| **Game over key** | Boolean under `"game_over"` or `"done"` | UNVERIFIED — capture required |
| **Rejection signalling** | Explicit `"result": "rejected"` or unchanged state | UNVERIFIED — capture required |
| **Step counter advancement** | Does a rejected move advance server step? | UNVERIFIED — capture required (Managed via `StepLedger`) |

*Actual Captured Fixture:* Pending execution of `python tools/probe_api.py --game snake --action start`.

---

## 5. 2048 State Schema

| Field / Aspect | Specification / Questions | Verified Status |
|---|---|---|
| **Board representation** | 4×4 nested integer list `[[int, ...], ...]` | UNVERIFIED — capture required |
| **Empty cell encoding** | `0` denotes empty tile | UNVERIFIED — capture required |
| **Score key** | Integer under `"score"` | UNVERIFIED — capture required |
| **Best tile** | Integer under `"max_tile"` / `"best_tile"` or `max(board)` | UNVERIFIED — capture required |
| **Spawn identification** | Delta comparison of `board_before` vs `board_after` | UNVERIFIED — capture required |
| **Rejection signalling** | Board identical after move or `"result": "rejected"` | UNVERIFIED — capture required |

*Actual Captured Fixture:* Pending execution of `python tools/probe_api.py --game 2048 --action start`.

---

## 6. Step Semantics

Per Genesis 2.0 Track EVA 02 specifications:
- The server counts valid transitions as accepted steps.
- Max accepted steps for Snake = **300**.
- Max accepted steps for 2048 = **2000**.
- Rejections (moving into an obstacle or wall where state doesn't transition) must be logged but tracked accurately so the accepted sequence remains gapless.

---

## 7. Rejected Step Policy

The Genesis 2.0 PDF requires:
> *"One line per APPLIED move, AND one line per REJECTED move... step starts at 1 with no gaps across applied sequence."*

To ensure compliance with organizer grading scripts:
- **Policy A (`"repeat_last_applied"`, Default):**
  - Applied moves advance `step`: `1, 2, 3, ...`
  - A rejected move logs `step = last_applied_step` (or `1` if rejected on step 1) and does NOT increment the counter.
  - Guarantees the applied sequence has zero gaps and matches final step count.
- **Policy B (`"pending_ordinal"`):**
  - Rejected move logs `step = next_step` but does not consume the ordinal.

---

## 8. Rate Limits, Retry & Timeout

- **Client Timeout:** `10.0` seconds default per HTTP request.
- **Retry Strategy:** Max 3 retries with exponential backoff factor `0.3s`.
- **Status Codes Retried:** `[429, 500, 502, 503, 504]`.
- **Method Scope:** Only idempotent GET and safe POST retried on transient errors.

---

## 9. Explicit Unknowns & Blockers

1. **TopGameAI API Documentation & Base URL:** Exact live production host and OpenAPI route unconfirmed.
2. **Endpoint Path Naming:** Whether paths use `/api/arena/{game}/` or `/arena/{game}/` or top-level `/snake/`.
3. **Payload Structure:** Exact field name for agent identification (`"agent"` vs `"agent_name"`).
4. **Laya Pip Package / Inference API:** Official client SDK or hosted endpoint documentation needed to swap out `MockLayaBackend`.
