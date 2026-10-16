# 🏆 Team ProgrammablePros: Master Project Briefing, Technical Architecture & Qwen/Judge AI Knowledge Base

**Competition:** Genesis 2.0 Hackathon | **Track:** EVA 02  
**Problem Statement:** Build an autonomous decision layer for hosted online games on TopGameAI Arena (Stage 1: Snake, Stage 2: 2048) using the **Laya** calibrated probability decision model under strict cryptographic audit constraints.  
**Repository:** `ProgrammablePros` (Python 3.10+)

---

## 1. Executive Summary & Core Philosophy

Most AI agents playing games rely on heavy generative LLMs (e.g. GPT-4, Claude, or general conversational checkpoints) that:
1. Suffer from high inference latency (1–3 seconds per turn), violating arena timing limits.
2. Cost dollars per session.
3. Frequently hallucinate non-existent directions or violate board geometry.
4. Fail completely under strict cryptographic audit constraints.

**ProgrammablePros takes an entirely different architectural path:**
- **Zero Hallucination via Bounded Hypotheses:** Instead of asking *"What should I do next?"*, the state is compressed into high-density tokens, transformed into **orthogonal typed scientific hypotheses**, and evaluated by **Laya** (a lightweight calibrated probability model) for pure mathematical success likelihoods ($0.0 \le P \le 1.0$) in milliseconds.
- **Expected Value (EV) Decision Engine:** The bot never blindly executes an `argmax(P)` policy. It evaluates multi-objective payoff matrices incorporating progress, structural advantage, severe danger penalties, and repeat move penalties.
- **Strict Compliance & Cryptographic Audit Chains:** Every move is bound into an immutable **SHA-256 hash-chain** JSON Lines log, guaranteed 4-way probability normalization summing to $1.000000 \in [0.98, 1.02]$, and capped at $\le 10\%$ safety fallback.
- **Offline & Hosted Parity:** Full decision brains, game engines, and live Streamlit dashboards work both in-process locally and over remote TopGameAI arena endpoints with zero code refactoring.

---

## 2. Official Competition Game Matrix

From the Genesis 2.0 Track EVA 02 official rules:
*"You do not build the games. The games run on the TopGameAI Arena site, and you never run them yourself."*  
For local development and continuous integration without live internet access, we maintain a 100% compliant **Local Test Harness** (`snake_game.py` and `game2048.py`).

| Game Environment | Grid / State | Key Rules & Constraints | Objective & Actions | Scoring / Points |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Snake**<br>*(Hosted: `LocalSnakeClient`<br>Local Harness: `snake_game.py`)* | 20x20 Grid | - 180° reversal strictly illegal (rejected)<br>- Wall/self-collision ends game<br>- Max 300 accepted moves | Eat 15 foods to win.<br>Actions: `UP`, `DOWN`, `LEFT`, `RIGHT` | $\text{Score} = \text{foods} \times 60 (+100 \text{ if win})$, capped at 1000.<br>**Unlock Stage 2:** $\ge 600$ points (10+ foods). |
| **Stage 2: 2048**<br>*(Hosted: `LocalGame2048Client`<br>Local Harness: `game2048.py`)* | 4x4 Grid | - Merges equal adjacent tiles once/move<br>- Spawn 2 (90%) or 4 (10%)<br>- Unchanged board is rejected<br>- Max 2000 accepted moves | Merge toward 512, 1024, 2048 tile.<br>Actions: `UP`, `DOWN`, `LEFT`, `RIGHT` | $\text{Points} = \min(1000, \max(0, (\log_2(\text{best}) - 6) \times 143 + \lfloor\frac{\text{score}}{1000}\rfloor))$.<br>**Unlock Stage 3:** Reach 512 tile.<br>**Judging:** Median of 3 best runs. |

---

## 3. High-Level Decision Architecture

The architecture is divided into clear functional pipelines:

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                               1. SENSE & COMPRESSION                                   │
 │   Raw Game State ──► SnakeCompressor / Board2048Compressor (<= 200 char token budget)  │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                               2. REASON & HYPOTHESIZE                                  │
 │   SubgoalManager (SURVIVE vs CHASE_FOOD / SURVIVE vs KEEP_CORNER)                      │
 │   HypothesisBuilder (8 typed questions: Q1 progress toward subgoal, Q2 hazard/danger)  │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                               3. INFER & CALIBRATE                                     │
 │   Laya Backend (Mock / Local / Http REST API) ──► Raw Likelihoods [0.0, 1.0]           │
 │   CalibrationEngine (Empirical Platt Scaling + Brier Score Tracking)                   │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                               4. DECIDE & OPTIMIZE                                     │
 │   SnakeEVCalculator / Board2048EVCalculator (Risk-Reward Argmax over EV)               │
 │   Legality Mask (Anti-reversal, Wall collision, Board unchanged check)                 │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         5. COMPLY, LOG & STREAM TELEMETRY                              │
 │   ComplianceLayer: Normalize 4-way probabilities (sum == 1.000000), <=10% fallback     │
 │   OfficialLogger: SHA-256 Hash Chain: H_n = SHA256(H_{n-1} | canonical_json)           │
 │   Streamlit Live Dashboard (viewer_local.py): Real-time UI, board matrix & audits      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep-Dive: Core Components & Strategies

### Strategy 1: State Compression & Token Budget Enforcement
- **Components:** `SnakeCompressor` (`snake_brain.py`), `Board2048Compressor` (`game2048_brain.py`).
- **Mechanism:** Strips bloated JSON structures into compact, high signal-to-noise strings under 200 characters:
  - **Snake Context:** `H:(x,y) D:dir L:len F:(fx,fy) dF:dist W:up/down/left/right B:ahead/left/right E:foods S:steps_rem`
  - **2048 Context:** `T:max_tile(2^log2)@corner(r,c) E:empty_count Sm:smoothness M:monotonicity Sc:score S:steps_rem`
- **Why it matters:** Generative and small parameter models degrade when flooded with nested UI metadata. Distilling invariants ensures the attention budget focuses 100% on spatial decision features.

### Strategy 2: Hypothesis-Driven "Typed Questions"
- **Components:** `SnakeHypothesisBuilder`, `Board2048HypothesisBuilder`.
- **Mechanism:** For each candidate action $a \in \{\text{up}, \text{down}, \text{left}, \text{right}\}$, generates two orthogonal questions (8 total per step):
  - **Q1 (Progress):** *"Context: {C}. Hypothesis: Action '{a}' is legal and progresses toward {subgoal}. What is the calibrated probability (0.0 to 1.0) that this is true?"*
  - **Q2 (Danger / Collision):** *"Context: {C}. Hypothesis: Action '{a}' causes immediate collision / board paralysis risking game over. What is the calibrated probability (0.0 to 1.0) that this is true?"*

### Strategy 3: Multi-Objective Expected Value (EV) Formulation
- **Components:** `SnakeEVCalculator`, `Board2048EVCalculator`.
- **Formulation:**
  - **Snake:**
    $$\text{EV}(a) = \big(P_{\text{progress}}(a) \times 10.0\big) - \big(P_{\text{danger}}(a) \times 40.0\big) - \text{penalty}_{\text{repeat}}(a) - \text{penalty}_{\text{illegal}}(a)$$
  - **2048:**
    $$\text{EV}(a) = \big(P_{\text{progress}}(a) \times 12.0\big) + \big(P_{\text{structure}}(a) \times 4.0\big) - \big(P_{\text{danger}}(a) \times 30.0\big) - \text{penalty}_{\text{repeat}}(a) - \text{penalty}_{\text{illegal}}(a)$$
- **Why it beats Argmax:** Naive argmax picks a move with $P=0.72$ over $P=0.70$. But if the first move carries a catastrophic wall crash penalty ($-100$), its EV is deeply negative, whereas the safer move yields high positive EV. EV mathematical argmax completely eliminates high-confidence suicide moves.

### Strategy 4: Failure-Adaptive Memory & In-Context Warnings
- **Components:** `LayaAgent.recent_outcomes` (`laya_agent.py`), `SnakeCompressor.record_failure()`, `Board2048Compressor.record_failure()`.
- **Mechanism:** When an attempted move results in zero progress or is rejected, the failure is recorded. On the next step, a dynamic warning is injected:
  `"WARNING: In the previous turn, taking action '{action}' resulted in no progress. Adjust probability accordingly."`
- **Result:** Drops the model's confidence for that action (e.g. $80\% \to 10\%$), penalizing its EV and automatically breaking oscillatory trap loops.

### Strategy 5: Compliance Layer & Strict PDF Alignment
- **Components:** `ComplianceLayer` (`compliance_layer.py`), `OfficialLogger` (`official_logger.py`).
- **Rules Enforced (from Genesis 2.0 Track EVA 02):**
  1. **Canonical JSON:** Sorted keys, no spaces (`separators=(',', ':')`), UTF-8, strict rejection of `NaN` or `Infinity`.
  2. **SHA-256 Hash Chain:**
     $$H_0 = \text{"GENESIS-" } + \text{session\_id}$$
     $$H_n = \text{SHA-256}\big(H_{n-1} + "|" + \text{canonical\_json}(\text{line without hash})\big)$$
  3. **Strict Probability Normalization:**
     Every logged decision must supply four keys: `{"up", "down", "left", "right"}` summing exactly to $1.000000 \in [0.98, 1.02]$. Residual rounding errors are absorbed into the dominant probability.
  4. **Strict Safety Fallback Budget:**
     Safety fallbacks (e.g., intercepting an illegal 180° reversal in Snake) must constitute $\le 10\%$ of all session moves. In our test benchmarks, fallback is maintained at **$0.00\%$** because the EV calculator's legality mask naturally prevents illegal moves.
  5. **Exact Integrity Invariants:**
     $$\text{confidence} \equiv \text{probabilities}[\text{chosen}] \quad \text{and} \quad \text{chosen} \equiv \text{action}$$

---

## 5. File Inventory & Roles

| File Path | Primary Class / Function | Purpose |
| :--- | :--- | :--- |
| `snake_brain.py` | `SnakeCompressor`, `SnakeSubgoalManager`, `SnakeHypothesisBuilder`, `SnakeEVCalculator` | Stage 1: Perception $\to$ Reasoning $\to$ EV decision pipeline for Snake. |
| `game2048_brain.py` | `Board2048Compressor`, `Board2048SubgoalManager`, `Board2048HypothesisBuilder`, `Board2048EVCalculator` | Stage 2: Perception $\to$ Reasoning $\to$ EV decision pipeline for 2048. |
| `compliance_layer.py` | `ComplianceLayer` | PDF compliance bridge, probability normalization, fallback checks, decision block builder. |
| `official_logger.py` | `OfficialLogger` | Creates append-only JSONL logs with cryptographic SHA-256 hash chains. |
| `run_local_games.py` | `run_session()`, `main()` | Unified CLI runner for local Snake and 2048 games with hash chain logging. |
| `viewer_local.py` | Streamlit Dashboard | Visual live viewer for Snake (20x20 emojis) and 2048 (styled CSS tiles). |
| `local_game_clients.py` | `LocalSnakeClient`, `LocalGame2048Client` | Mock-arena client wrappers mimicking hosted REST endpoints. |
| `snake_game.py` | `SnakeGame` | Local test harness: 20x20 Snake simulation (PS2 §4.1). |
| `game2048.py` | `Game2048` | Local test harness: 4x4 2048 simulation with official points formula (PS2 §4.2). |
| `game_clients.py` | `SnakeClient`, `Game2048Client` | Live hosted TopGameAI API clients for competition arena. |
| `orchestrator_compliance.py` | `ComplianceOrchestrator` | Hosted arena runner enforcing stage gates and official logging. |
| `laya_agent.py` | `LayaAgent` | Interface for Laya inference, question formulation, and outcome memory. |
| `calibration_engine.py` | `CalibrationEngine` | Post-hoc Platt scaling and Brier score tracking. |

---

## 6. Verified Benchmark Results

### 1. Snake Local Benchmark
```bash
python run_local_games.py --game snake --seed 42 --use-mock-laya
```
*Result:*
- **Foods Eaten:** 15 / 15 (MAX WIN)
- **Score:** 1000 / 1000
- **Steps:** 231 accepted moves
- **Fallback Rate:** 0.00% (Budget $\le 10\%$)
- **Hash Chain:** `Hash chain VALID — 231 entries verified.`

### 2. 2048 Local Benchmark
```bash
python run_local_games.py --game 2048 --seed 42 --use-mock-laya
```
*Result:*
- **Best Tile:** 256 (Progressing toward 512/1024)
- **Score:** 3548
- **Official Points:** 289
- **Steps:** 302 accepted moves
- **Fallback Rate:** 0.00% (Budget $\le 10\%$)
- **Hash Chain:** `Hash chain VALID — 302 entries verified.`

---

## 7. Judge & AI Model Defense Briefing (Q&A)

### Q1: Why not just use GPT-4o or Claude 3.5 Sonnet to play the games?
> **Answer:** Large generative models have three fatal flaws in competitive gaming:
> 1. **Latency:** 1–3 seconds per turn violates arena response windows. Laya runs in $\le 20\text{ms}$.
> 2. **Hallucination:** Generative models invent invalid actions or drift from formatting requirements. Laya outputs bounded likelihoods for strict typed questions.
> 3. **Cost & Non-Determinism:** Frontier LLMs cost dollars per game and fail cryptographic reproducibility. Laya paired with our EV calculator produces deterministic, auditable decisions.

### Q2: What is the primary difference between your bot and an `argmax(P)` bot?
> **Answer:** An `argmax` bot blindly executes whichever move has the highest raw probability. In complex games, payoffs are asymmetric. A move with 72% success that risks a wall crash ($-100$ penalty) yields an EV of deeply negative expectation. A move with 70% success and standard penalty yields positive EV. Our EV engine calculates mathematical expectation, completely avoiding suicidal high-probability traps.

### Q3: How do you guarantee the official logs are authentic and not tampered with?
> **Answer:** We enforce an in-process cryptographic SHA-256 hash chain on every single move. Each line contains `prev_hash` and `hash = SHA256(prev_hash | canonical_json)`. If a single character, probability decimal, or step number is altered post-hoc, `OfficialLogger.verify_log_file()` immediately flags the chain break.

### Q4: How easily does this codebase switch to the live hackathon arena?
> **Answer:** Seamlessly in under 2 minutes. The core decision brains (`snake_brain.py`, `game2048_brain.py`), compression layers, EV calculators, and compliance logger remain 100% untouched. All that changes is switching from `LocalSnakeClient` / `LocalGame2048Client` to the hosted arena REST endpoints and updating `LAYA_MODE=http` in `.env`.
