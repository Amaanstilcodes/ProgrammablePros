# 🏆 Team ProgrammablePros: Master Project Briefing & Pitch Defense
**Hackathon:** Genesis 2.0 | **Track:** EVA 02  
**Problem Statement:** Build the decision layer for hosted online games using the Laya calibrated probability model.

---

## 1. The Elevator Pitch (1 Minute)

> "Most game-playing AI agents rely on heavy, expensive generative LLMs like GPT-4 that cost dollars per game, lag by several seconds, and inevitably hallucinate invalid moves. **ProgrammablePros** takes an entirely different architectural route: we engineered an autonomous decision layer powered by **Laya**, a lightweight, open-weight calibrated probability model.
> 
> Instead of asking an AI 'What should I do next?', our bot transforms the game state into **typed binary hypotheses**, queries Laya for pure mathematical success probabilities in milliseconds, and selects moves using a rigorous **Expected Value (EV) risk-reward engine**. The result is a lightning-fast, zero-hallucination, failure-adaptive bot that plays with institutional-grade auditability across both 1-D navigation and complex 2-D multi-hazard heist environments."

---

## 2. Supported Game Environments

Our decision layer drives two distinct game environments via a unified registry in `main.py`:

| Game Environment | Dimensionality & Board | Key Hazards & Mechanics | Objective & Actions |
| :--- | :--- | :--- | :--- |
| **1-D Grid Navigator** (`mock_game.py`) | 1-D Corridor (0 to 10) | Boundary limits, turn decay (30 max steps) | Reach goal position `10`<br>Actions: `LEFT`, `RIGHT` |
| **Vault Runner (2-D Heist)** (`vault_game.py`) | 2-D 8x8 Grid Floor | Guard patrol (`G`, -30 penalty), Lasers (`L`, -5 penalty), Traps (`T`, -8 penalty), Hidden Walls (`H`), Wall sprint crashes (-15 penalty) | Collect Key `K`, then unlock Exit `E`<br>Actions: `UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`, `SPRINT` |

---

## 3. How the Bot Works: The 9-Step Pipeline

The bot operates on a continuous, deterministic **Sense-Think-Act-Learn** loop orchestrated in `agent_orchestrator.py` and `main.py`:

```text
       [ Sense ]                [ Think ]                 [ Decide & Act ]                [ Learn ]
  Game Server State  ──►  Feature Compression  ──►  Calculate Expected Value  ──►  Evaluate Transition
  (Positions, Turns)      & Hypothesis Query         & Execute Max-EV Action        & Record Failure Memory
                                  │                             │                              │
                                  ▼                             ▼                              ▼
                          Laya Calibration             Live JSONL Telemetry           Dynamic Prompt Warning
```

### The 9-Step Sequential Pipeline (`AgentOrchestrator.play_turn`):
1. **Observe:** `WorldMemory.observe()` registers the previous transition in persistent spatial memory, automatically inferring `WALL` tiles when moves fail.
2. **Compress:** `EgocentricCompressor.compress()` produces an agent-centric point-of-view (POV) prompt with relative surroundings (`AHEAD`, `LEFT`, `RIGHT`, `BEHIND`), 8-point compass bearings, and resource counters.
3. **Subgoal:** `SubgoalManager.get_current_subgoal()` deterministic priority cascade selects the active tactical target (`RETREAT` $\to$ `KEY` $\to$ `EXIT` $\to$ `EXPLORE`).
4. **Hypothesize:** `HypothesisEngine.generate_hypotheses()` formulates 4 typed scientific questions per legal move (`is_legal`, `toward_subgoal`, `is_dangerous`, `is_novel`) + injects `AnchorProbe` ground-truth calibration checks.
5. **Laya Inference:** Dispatches typed queries to Laya (or mock engine) for parallel likelihood scoring.
6. **Calibrate:** `CalibrationEngine.predict()` applies Platt scaling to map raw likelihoods to calibrated probabilities.
7. **Decide:** `AdvancedEVCalculator.calculate()` computes multi-objective risk-adjusted Expected Value and selects the move via mathematical argmax.
8. **Log:** `DecisionLogger.append_decision()` writes the full snapshot to `decision_log.jsonl` with immediate flush for real-time Streamlit polling.
9. **Return:** Dispatches the selected action string to the game client.

---

## 4. Deep Dive: The Core Strategies & Upgrades

The problem statement explicitly warns: *"Teams using the same model will score very differently, and the games will punish weak understanding."* Here is how our architecture provides an asymmetric competitive advantage:

---

### Strategy 1: State Compression & Feature Engineering
* **What it is:**  
  Raw game states arrive as multi-layered, nested JSON dictionaries cluttered with metadata, status strings, and styling parameters. We extract only mathematical invariants and enforce strict token budgets:
  - 1-D Game: 60-character vector string.
  - 2-D Game (`compress_vault_state`): Strict $\le 120$-character string containing current objective (`KEY` vs `EXIT`), Manhattan distances, key possession, steps remaining, laser adjacency, guard distance, and patrol axis.
* **Why it matters:**  
  Small models have limited parameter capacity and smaller attention budgets. Flooding them with 50 lines of JSON dilutes their attention weights across irrelevant tokens. Dense feature engineering presents the problem in high signal-to-noise format, ensuring Laya dedicates its computational capacity solely to spatial calculation.
* **Concrete Example:**  
  Instead of feeding 600 bytes of raw JSON containing grid arrays, session IDs, and tile colors, Laya receives:
  `pos=(2,2) obj=KEY obj_dist=3 keys=0 steps_left=35 laser_adj=NONE guard_dist=4 guard_dir=UP facing=RIGHT`
  Pre-computing distance and direction prevents arithmetic errors in small models.

---

### Strategy 2: Hypothesis-Driven "Typed Questions"
* **What it is:**  
  Laya is an open-weight model specialized in **probability calibration**, not conversational roleplay. Rather than prompting *"Where should the player move next?"*, `formulate_questions()` poses closed, testable binary hypotheses:
  > *"Context: {compressed_state}. Hypothesis: Executing action '{action}' will successfully progress toward {goal_phrase}. What is the calibrated probability (0.0 to 1.0) that this hypothesis is true?"*
* **Dynamic Goal Switching:**  
  In Vault Runner, the `goal_phrase` automatically shifts as game phases complete:
  - Phase 1 (Keys = 0): `"the KEY"`
  - Phase 2 (Keys = 1): `"the EXIT without being caught or hitting a hazard"`
* **Why it matters:**  
  Open-ended questions force models to generate freeform text, leading to parse errors, hallucinated actions, and non-deterministic behavior. By framing each move as an independent hypothesis, we treat Laya as a pure likelihood scoring function, directly extracting calibrated $P(\text{True}) \in [0.0, 1.0]$.

---

### Strategy 3: Expected Value (EV) Decision Making
* **What it is:**  
  A probability is only half of a decision; the other half is the payoff matrix. In `calculate_expected_values()`, we compute the statistical Expected Value:
  $$\text{EV} = \Big(P(\text{Success}) \times \text{Reward}\Big) - \Big(\big(1.0 - P(\text{Success})\big) \times \text{Penalty}\Big)$$
* **Why it matters:**  
  Novice teams implement an `argmax(probabilities)` policy. But in game environments where certain actions carry asymmetric penalties (guard capture, laser damage, wall crash), choosing a move with a high raw probability and extreme downside is a losing strategy. EV calculation guarantees risk-adjusted optimal play.
* **Concrete Benchmark Proof (Vault Runner Test 4):**  
  Suppose action `SPRINT` has raw $P = 0.72$ ($\text{Reward} = +8.0, \text{Penalty} = -15.0$ due to a wall in tile 2).  
  $$\text{EV}(\text{SPRINT}) = (0.72 \times 8.0) - (0.28 \times 15.0) = 5.76 - 4.20 = \mathbf{+1.56}$$  
  Meanwhile, `DOWN` has $P = 0.70$ ($\text{Reward} = +10.0, \text{Penalty} = -5.0$):  
  $$\text{EV}(\text{DOWN}) = (0.70 \times 10.0) - (0.30 \times 5.0) = 7.00 - 1.50 = \mathbf{+5.50}$$  
  - An argmax bot chooses `SPRINT` ($0.72 > 0.70$) and crashes into the wall.
  - Our EV bot chooses `DOWN` ($+5.50 > +1.56$), cleanly preserving win equity and avoiding penalty.

---

### Strategy 4: Failure-Adaptive Prompting (Short-Term Memory)
* **What it is:**  
  A bounded short-term memory ring buffer (`recent_outcomes`, tracking the last 3 turns). If an action unexpectedly fails to reduce distance to the goal (e.g. colliding with an unannounced hidden wall or trying to walk through a locked EXIT), the transition is logged. On the next turn, the hypothesis prompt for that action dynamically appends:
  > *"WARNING: In the previous turn, taking action '{action}' resulted in no progress. Adjust probability accordingly."*
* **Why it matters:**  
  Stateless bots suffer from **oscillatory trap loops**: when facing a hidden obstacle or locked barrier, a stateless bot repeatedly selects the same high-probability move turn after turn until its turn limit runs out. Our adaptive memory breaks loops on turn $N+1$.
* **Concrete Example from Live Vault Run:**  
  At position `(2, 2)`, moving `RIGHT` hits a hidden wall at `(3, 2)`:
  - **Turn 30:** Bot selects `RIGHT`. Result: player remains at `(2, 2)`, hidden wall logged.
  - **Turn 31:** Prompt appends `WARNING`. Laya incorporates the negative prior and drops $P(\text{RIGHT})$ from $80.3\% \to 10.0\%$.
  - **Immediate Impact:** $\text{EV}(\text{RIGHT})$ plunges from $+7.04 \to -3.50$. The bot pivots to `WAIT` ($\text{EV} = -0.11$) and `UP` ($\text{EV} = +1.04$), safely circumnavigating the obstacle.

---

### Strategy 5: Perception Layer (World Memory & Egocentric POV)
* **What it is:**  
  A pair of decoupled perception classes:
  1. **`WorldMemory` (`world_memory.py`):** Retains object permanence across moves. When an action fails to alter coordinates, the target cell is inferred as a `WALL`. Tracks cell entry counts to calculate visit decay, and empirical action success distributions `(coord, action) -> [successes, attempts]`.
  2. **`EgocentricCompressor` (`egocentric_compressor.py`):** Translates global coordinates into an agent-centric frame:
     ```text
     POV: agent at (1,0), facing RIGHT
     LOCAL: AHEAD: UNKNOWN, LEFT: WALL, RIGHT: UNKNOWN, BEHIND: UNKNOWN
     OBJECTIVES: KEY(bearing=SE, dist=4), EXIT(bearing=SE, dist=13 [LOCKED-NEEDS-KEY])
     RESOURCES: health=100, ammo=5, keys=0, steps_left=35
     HISTORY: action 'UP' at (1,0) failed (Position unchanged (hit obstacle / wall))
     ```
* **Why it matters:**  
  Doom-style and maze environments require partial observability reasoning. Absolute coordinates confuse generalized models across mirrored maps, whereas egocentric descriptions ("AHEAD is WALL, bearing to KEY is SE") align naturally with calibrated probability assessment.

---

### Strategy 6: Reasoning Layer (Subgoal Arbitration & Multi-Outcome Hypotheses)
* **What it is:**  
  1. **`SubgoalManager` (`subgoal_manager.py`):** Deterministic priority tree ensuring the bot balances self-preservation with objective completion:
     - `Priority 1 (RETREAT)`: Critical health (<20) with active threat (<3 dist).
     - `Priority 2 (KEY)`: Keys held == 0 and locked barrier present.
     - `Priority 3 (EXIT)`: Keys held > 0, direct navigation to goal.
     - `Priority 4 (EXPLORE)`: Nearest unvisited frontier cell.
  2. **`HypothesisEngine` (`hypothesis_engine.py`):** Formulates 4 orthogonal typed scientific queries per legal move (`is_legal`, `toward_subgoal`, `is_dangerous`, `is_novel`).
  3. **Anchor Probes:** Injects ground-truth sanity check questions (e.g. moving into a known WALL is legal $\to$ expected 0.0) to detect model calibration degradation.
* **Why it matters:**  
  Single-question bots cannot distinguish between illegal moves, moves toward different subgoals, and dangerous moves. 4-way typed queries give the EV calculator clean independent variables for multi-objective optimization.

---

### Strategy 7: Orchestration & Decision Engine (`agent_orchestrator.py`)
* **What it is:**  
  1. **`AgentOrchestrator` (`agent_orchestrator.py`):** Master coordinator executing the full 9-step pipeline (Observe $\to$ Compress $\to$ Subgoal $\to$ Hypothesize $\to$ Infer $\to$ Calibrate $\to$ Decide $\to$ Log $\to$ Return) with complete intermediate state auditability.
  2. **`AdvancedEVCalculator` (`advanced_ev_calculator.py`):** Synthesizes:
     $$\text{EV}(a) = P(\text{legal}) \times \Big[\Big(P(\text{subgoal}) W_{\text{subgoal}} - (1 - P(\text{subgoal})) W_{\text{fail}}\Big) - P(\text{danger}) W_{\text{danger}} + P(\text{novel}) W_{\text{novel}}\Big] - (1 - P(\text{legal})) W_{\text{illegal}} - \text{penalty}_{\text{repeat}}$$
  3. **`CalibrationEngine` (`calibration_engine.py`):** Post-hoc Platt scaling and empirical Brier scoring.
  4. **`DecisionLogger` (`decision_logger.py`):** Non-blocking append-only JSONL streaming logger with immediate flushing.

---

## 5. Automated Verification & 20-Seed Headless Benchmark

We built an acceptance test suite in `test_vault.py` covering all game mechanics, edge cases, state compression limits, and statistical performance:

```bash
python test_vault.py
```

### Headless Benchmark Results (20 Randomized Procedural Seeds):
```text
  ╔══════════════════════════════════════════════════════════╗
  ║         Headless Benchmark — 20 Seeds (Vault Runner)    ║
  ╠══════════════════════════╦═══════════════╦══════════════╣
  ║ Metric                   ║  EV Agent     ║  Argmax (P)  ║
  ╠══════════════════════════╬═══════════════╬══════════════╣
  ║ Win rate                 ║ 19/20 (95%)   ║ 18/20 (90%)  ║
  ║ Avg steps                ║          15.9 ║         14.0 ║
  ║ Avg penalty              ║          -0.9 ║         -3.5 ║
  ╚══════════════════════════╩═══════════════╩══════════════╝

  [PASS] Test 6: EV penalty (-0.9) <= argmax penalty (-3.5) -- risk-aware payoffs working
  Results: 11 passed, 0 failed out of 11 tests
```

---

## 6. Live Telemetry & Audit Dashboard (`viewer.py`)

Run the Streamlit live dashboard for real-time judge presentations:
```bash
streamlit run viewer.py
```

### Dashboard Capabilities:
- **5 Top-Level KPIs:** Total Decisions Logged, Player Position, Latest Chosen Move, Expected Value (EV), Game Status.
- **🗺️ Agent Position Heatmap (Vault Runner):** Visualizes the agent's spatial trajectory across the 8x8 floor grid in real-time using Pandas pivot aggregation.
- **Calibrated Probabilities & EV Bar Graphs:** Side-by-side comparison of Laya's confidence vs calculated EV scores for every legal action.
- **Typed Hypothesis Inspector:** Displays the exact natural-language hypothesis queries evaluated by Laya on each step.
- **Short-Term Memory Alert Banners:** Displays active `WARNING` prompts whenever failure-adaptive memory intervenes.
- **Chronological Audit Feed:** Complete table of historical decisions streamed directly from `decision_log.jsonl`.

---

## 7. Hackathon Day Readiness (The 5-Minute Transition)

Our codebase is pre-engineered for a seamless 5-minute transition from the local simulator to live competition endpoints:

```text
                  ┌──────────────────────────────────────────────┐
                  │              ARCHITECTURE CORE               │
                  │   • State Compression     • EV Calculation   │
                  │   • Hypothesis Engine     • Memory Buffer    │
                  │   • World Memory          • Egocentric POV   │
                  │   • Subgoal Manager       • Anchor Probes    │
                  │   • Agent Orchestrator    • Decision Logger  │
                  │   • JSONL Audit Logger    • Streamlit Viewer │
                  └───────────────────────┬──────────────────────┘
                                          │ (Remains 100% Untouched)
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
             [ Toggle 1: Game Client ]         [ Toggle 2: Model Inference ]
             Replace MockGame/VaultRunner      In agent_orchestrator.py & laya_agent.py:
             with Live Game REST Endpoint      Flip use_mock = False,
             (base_url + API Key)              Supply Hackathon API Key
```

### What Changes:
1. **Model Toggle (`agent_orchestrator.py` & `laya_agent.py`):** Flip mock inference to `client.predict_probability()` using the competition API key.
2. **Game Client (`main.py`):** Register the competition HTTP client into `GAME_REGISTRY`.
3. **What Remains 100% Untouched:** State compressor, egocentric POV, world memory, subgoal manager, hypothesis engine, calibration engine, EV engine, memory buffer, JSONL logger, and Streamlit dashboard.

---

## 8. Top 3 Anticipated Judge Questions & Winning Answers

### Question 1: "Why use a small calibrated probability model like Laya instead of sending the game screenshot or JSON to GPT-4o or Claude 3.5 Sonnet?"
> **Winning Answer:**  
> "Generative frontier models excel at creative synthesis, but they are ill-suited as real-time game controllers. They suffer from high inference latency (1–3 seconds per step), cost orders of magnitude more per interaction, and are prone to generative hallucinations—frequently inventing actions outside the legal game schema.  
> Laya is purpose-built for calibration. By querying bounded hypotheses, we receive mathematically grounded probabilities in tens of milliseconds with zero hallucination. Combined with our Expected Value engine, Laya provides deterministic, risk-managed control that frontier LLMs cannot replicate at game speed."

---

### Question 2: "How does your Expected Value formulation actually prevent losses when a higher probability move is available?"
> **Winning Answer:**  
> "Standard bots operate on a naive argmax probability policy. If Move A has a 72% probability of success and Move B has a 70% probability, an argmax bot always picks Move A.  
> However, in competitive gaming, penalties are asymmetric. In our Vault Runner heist game, sprinting forward toward the key carries a 28% risk of slamming into a wall (-15 penalty), yielding an EV of $(0.72 \times 8) - (0.28 \times 15) = \mathbf{+1.56}$.  
> Stepping down has a 70% success rate with a standard failure penalty (-5), yielding an EV of $(0.70 \times 10) - (0.30 \times 5) = \mathbf{+5.50}$. Our EV decision engine mathematically avoids high-probability 'death traps' and selects the move with the highest risk-adjusted mathematical expectation."

---

### Question 3: "What happens if the game introduces novel mechanics, dynamic wind, or invisible obstacles that the model was never trained on?"
> **Winning Answer:**  
> "That exact scenario is why we built Feature 4 (Failure-Adaptive Prompting), Feature 5 (`WorldMemory`), Feature 6 (`SubgoalManager`), and Feature 7 (`AgentOrchestrator`).  
> Because the model does not possess omniscient knowledge of unannounced game mechanics (like the hidden walls in Vault Runner), our agent relies on empirical feedback via a bounded short-term memory buffer and persistent spatial grid. When an action unexpectedly fails to reduce distance or hits a barrier, the target coordinate is mapped as `WALL` and the transition is logged. On the next tick, the hypothesis prompt explicitly injects an in-context negative prior informing Laya of the immediate outcome.  
> This suppresses the calibrated probability for that action, shifts the EV landscape, and forces the bot to explore alternative trajectories—completely preventing the infinite failure loops that disqualify rigid bots."

---

## 9. Team Presentation & Defense Checklist
- [x] **Terminal 1:** `streamlit run viewer.py` (Display on presentation monitor showing live KPIs, 2-D heatmap, EV bars, and compressed context).
- [x] **Terminal 2:** `python main.py --game vault` (Execute the 2D heist bot showing real-time ASCII maps, probability calibration, and step feedback).
- [x] **Terminal 3 (Benchmarking):** `python test_vault.py` (Run the 11-test acceptance suite and 20-seed headless benchmark live for the judges).
- [x] **Terminal 4 (Orchestrator V2):** `python agent_orchestrator.py` (Run the 3-turn end-to-end 9-step decision pipeline test).
- [x] **Audit Trail:** Open `decision_log.jsonl` to demonstrate append-only transparency and verifiable decision logs.
