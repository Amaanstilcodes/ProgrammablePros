# 🤖 ProgrammablePros: Laya-Powered Autonomous Game Bot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon Track](https://img.shields.io/badge/Track-EVA%2002-green.svg)]()

> **Team Name:** ProgrammablePros  
> **Track:** EVA 02  
> **Problem Statement:** Build the decision layer for hosted online games using the Laya calibrated probability model.

---

## 📖 Project Overview

Most "AI plays a game" projects rely on large, slow, and expensive generative LLMs that hallucinate actions. This project takes a different approach. We built an autonomous bot that uses **Laya**, a small, open-weight calibrated probability decision model. 

Instead of asking an AI *"What should I do?"*, our bot translates the game state into **typed, binary hypothesis questions** (e.g., *"What is the calibrated probability that executing 'RIGHT' progresses toward the goal?"*). Laya returns fast, mathematically sound probabilities, which our bot evaluates using an **Expected Value (EV)** risk-reward calculator to make optimal, auditable decisions.

---

## 🚀 Our Unique Approach (The Core Challenge)

The problem statement notes: *"Teams using the same model will score very differently, and the games will punish weak understanding."* To win, we engineered advanced strategies directly into our decision layer:

1. **🧹 State Compression & Feature Engineering:** Small models get confused by noisy, raw JSON. We built compressors that strip away UI noise and feed Laya high-signal variables (e.g., `player_pos`, `goal_pos`, `distance_to_goal`, `steps_remaining`).
2. **🎯 Hypothesis-Driven "Typed Questions":** We frame prompts as strict, testable hypotheses rather than open-ended queries, forcing Laya to do what it does best: binary/multi-class probability calibration.
3. **⚖️ Expected Value (EV) Decision Making:** We don't blindly pick the highest probability. We calculate $EV = (P_{success} \times Reward) - (P_{failure} \times Penalty)$ to weigh the risk vs. reward of every move, preventing catastrophic failures.
4. **🧠 Failure-Adaptive Prompting (Short-Term Memory):** If a move results in no progress, the bot logs this outcome. On the next turn, it dynamically appends a `WARNING` to Laya's prompt for that specific action, forcing the model to adjust its probability downward and triggering an immediate course correction.
5. **👁️ Perception Layer & Egocentric World Memory (V2 Upgrade):**
   - **`WorldMemory` (`world_memory.py`):** Maintains persistent spatial knowledge (`cells: (x,y) -> WALL | FREE | UNKNOWN`), entry visit counts, and empirical outcome rates (`outcomes: ((x,y), action) -> [successes, attempts]`). Automatically marks obstacles as `WALL` when movement attempts leave coordinates unchanged.
   - **`EgocentricCompressor` (`egocentric_compressor.py`):** Transforms global states into an agent-centric point-of-view (POV) representation with relative neighbor states (`AHEAD`, `LEFT`, `RIGHT`, `BEHIND`), 8-point compass bearings (`N`, `NE`, `E`, etc.), Manhattan distances, resource tallies, and recent failure histories.
6. **🎯 Reasoning Layer & Multi-Outcome Hypotheses (V2 Upgrade):**
   - **`SubgoalManager` (`subgoal_manager.py`):** Arbitrates hierarchical objectives with deterministic priorities: `RETREAT` (critical health & threat in range) $\to$ `KEY` (need key to unlock door) $\to$ `EXIT` (key secured, proceed to escape) $\to$ `EXPLORE` (nearest unexplored frontier).
   - **`HypothesisEngine` (`hypothesis_engine.py`):** Dispatches 4 typed hypothesis questions per legal action (`is_legal`, `toward_subgoal`, `is_dangerous`, `is_novel`) and injects ground-truth `AnchorProbe` questions to monitor model calibration drift in real time.
7. **⚙️ Orchestration & Decision Engine (V2 Upgrade):**
   - **`AgentOrchestrator` (`agent_orchestrator.py`):** Executes the full 9-step Sense-Compress-Subgoal-Hypothesize-Infer-Calibrate-Decide-Log-Act pipeline end-to-end.
   - **`AdvancedEVCalculator` (`advanced_ev_calculator.py`):** Synthesizes legality filters, subgoal utility, danger penalties, novelty bonuses, and repeat failure suppression.
   - **`CalibrationEngine` (`calibration_engine.py`):** Post-hoc Platt scaling and empirical Brier score tracking.
   - **`DecisionLogger` (`decision_logger.py`):** Non-blocking append-only JSONL streaming logger with immediate flushing for live telemetry.

---

## 🏗️ Architecture & Tech Stack

| Technology | Role in Project | Engineering Justification |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Runtime & Logic | Native support for modern typing, async capabilities, and ML SDKs. |
| **Laya Inference** | Decision Intelligence | Evaluates typed hypotheses and returns calibrated probabilities ($0.0 \le P \le 1.0$), preventing generative hallucinations. |
| **Perception Layer** | Spatial Memory & Egocentric POV | `WorldMemory` + `EgocentricCompressor` provide object permanence and relative bearing perception without hardcoded pathfinding. |
| **Reasoning Layer** | Subgoals & 4-Way Hypotheses | `SubgoalManager` + `HypothesisEngine` drive hierarchical goal arbitration and multi-objective query calibration. |
| **Decision Engine** | Multi-Objective EV & Calibration | `AdvancedEVCalculator` + `CalibrationEngine` synthesize 4-way utility and post-hoc Platt scaling. |
| **Master Orchestrator** | 9-Step Pipeline Execution | `AgentOrchestrator` integrates perception, reasoning, calibration, and action selection. |
| **JSON Lines (JSONL)** | Audit Trail & Logging | `DecisionLogger` streaming format that eliminates file locks, allowing concurrent reading by the dashboard while the bot writes. |
| **Streamlit** | Live Telemetry Dashboard | Zero-friction, reactive web UI for real-time monitoring of bot confidence, EV scores, and audit records. |

### Project Structure
```text
ProgrammablePros/
├── agent_orchestrator.py   # Agent V2: Master orchestrator running the 9-step decision pipeline
├── decision_logger.py      # Agent V2: Append-only JSONL streaming logger with immediate flushing
├── advanced_ev_calculator.py# Agent V2: Multi-objective EV engine (subgoal/danger/novelty/repeat)
├── calibration_engine.py   # Agent V2: Platt scaling calibration and empirical Brier scoring
├── subgoal_manager.py      # Agent V2: Deterministic priority hierarchy (RETREAT -> KEY -> EXIT -> EXPLORE)
├── hypothesis_engine.py    # Agent V2: 4-way typed queries (legal/subgoal/danger/novel) + anchor probes
├── world_memory.py         # Agent V2: Persistent spatial memory, visit tracking, & outcome stats
├── egocentric_compressor.py# Agent V2: POV egocentric state compressor (AHEAD/LEFT/RIGHT, bearing, dist)
├── main.py                 # Unified game loop, CLI orchestrator, EV engine, & memory buffer
├── laya_agent.py           # State compression, hypothesis formulation, and Laya API integration
├── vault_game.py           # 2-D 8x8 Vault Runner heist game (guards, lasers, traps, hidden walls)
├── mock_game.py            # 1-D grid game environment (swapped for real API on hackathon day)
├── test_vault.py           # 11-test acceptance suite & 20-seed headless EV benchmark
├── logger.py               # Append-only JSONL decision logging utility (V1 baseline)
├── viewer.py               # Streamlit live telemetry dashboard with 2-D heatmap & EV bars
├── decision_log.jsonl      # Append-only live streaming audit log
├── requirements.txt        # Python dependencies
├── MASTER_EXPLANATION.md   # Pitch defense, architectural guide, & judge Q&A briefing (Markdown)
├── MASTER_EXPLANATION.txt  # Synchronized pitch defense & briefing (Plaintext)
└── README.md               # Documentation and run instructions
```

---

## ⚡ Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd ProgrammablePros
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*(Optional: Install `pandas` for the interactive 2-D agent heatmap in the Streamlit viewer: `pip install pandas`)*

---

## 🎮 Demo Games & How to Run Them

We provide **two full game environments**, an **EV vs. Argmax baseline comparison**, and a **20-seed headless benchmark suite** to demonstrate Laya's calibrated decision making across simple corridors and complex multi-hazard 2D heist scenarios.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                              DEMO RUN GUIDE                                │
├───────────────────────────────┬────────────────────────────────────────────┤
│ Scenario                      │ Command                                    │
├───────────────────────────────┼────────────────────────────────────────────┤
│ 1. Classic 1-D Navigator      │ python main.py                             │
│ 2. Vault Runner (2-D Heist)   │ python main.py --game vault                │
│ 3. Argmax Baseline (No EV)    │ python main.py --game vault --no-ev        │
│ 4. Acceptance & Benchmark     │ python test_vault.py                       │
│ 5. Agent V2 Pipeline Test     │ python agent_orchestrator.py               │
│ 6. Live Streamlit Dashboard   │ streamlit run viewer.py                    │
└───────────────────────────────┴────────────────────────────────────────────┘
```

---

### Demo 1: Classic 1-D Grid Navigator

A focused demonstration of directional calibration along a 1-D linear corridor where the agent navigates from start position `0` to goal position `10` within 30 turns.

#### How to Run:
```bash
python main.py
```
*(or explicitly: `python main.py --game classic`)*

#### Active Strategies Demonstrated:
- **Strategy 1 (Compression):** Strips structural noise down to `pos=X, goal=10, dist=Y, steps=Z`.
- **Strategy 2 (Typed Questions):** Asks binary hypothesis queries for `LEFT` and `RIGHT`.
- **Strategy 3 (EV Engine):** Computes Expected Value using $(P \times +10.0) - ((1 - P) \times 5.0)$.
- **Strategy 4 (Short-Term Memory):** Prevents back-and-forth oscillation.

#### Expected Console Output & Results:
```text
======================================================================
[*] AI GAME BOT -- POWERED BY LAYA CALIBRATED PROBABILITIES [*]
======================================================================
  Game Mode:           1-D Grid Navigator
  Decision Model:      Laya (Calibrated Probability Scoring)
  Active Strategies:   1. State Compression & Feature Engineering
                       2. Hypothesis-Driven Typed Questions
                       3. Expected Value (EV) Risk/Reward Optimization
                       4. Failure-Adaptive Short-Term Memory
  Log Output:          decision_log.jsonl
  Live Viewer:         Run 'streamlit run viewer.py' in a separate terminal
======================================================================

[+] Initialising Game Session...
   Start Position: 0 | Target Goal: 10

[Step 01] Pos:  0/10 | Probs: ['left': 17.3%, 'right': 82.0%] | EVs: ['left': -2.40, 'right': +7.30] | -> Move: 'RIGHT' (EV: +7.30)
[Step 02] Pos:  1/10 | Probs: ['left': 11.9%, 'right': 73.2%] | EVs: ['left': -3.21, 'right': +5.98] | -> Move: 'RIGHT' (EV: +5.98)
[Step 03] Pos:  2/10 | Probs: ['left': 8.0%, 'right': 87.7%] | EVs: ['left': -3.80, 'right': +8.15] | -> Move: 'RIGHT' (EV: +8.15)
...
[Step 10] Pos:  9/10 | Probs: ['left': 20.4%, 'right': 87.1%] | EVs: ['left': -1.94, 'right': +8.06] | -> Move: 'RIGHT' (EV: +8.06)

======================================================================
[WIN] SUCCESS! Reached objective in 10 steps!
[*] All decisions with compressed states & EVs recorded in 'decision_log.jsonl'.
======================================================================
```
- **Outcome:** **100% Win Rate (10/10 optimal steps)** with zero wasted turns or backtracking.

---

### Demo 2: Vault Runner (2-D Heist Game)

A complex, multi-stage 8x8 heist environment requiring the bot to navigate from Start `S` (0,0), retrieve the KEY `K`, and escape through the EXIT `E` (7,7) while avoiding active hazards.

#### Game Mechanics & Grid Legend:
- `P` = Player Agent
- `G` = Guard patrolling back-and-forth along an axis (moving adjacent to or into guard carries severe **-30 penalty**)
- `K` = Vault Key (must be collected before EXIT unlocks)
- `E` = Vault Exit (locked until Key is held)
- `W` = Solid Walls (blocks movement)
- `H` = **Hidden Walls** (unannounced barriers invisible in visible grid; discovered upon collision)
- `L` = Laser Sensor (**-5 penalty** per step spent on laser)
- `T` = Trap Floor (**one-time -8 penalty**, converts to regular floor once triggered)
- **6 Legal Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT` (pause turn while guard passes), `SPRINT` (jump 2 tiles forward in facing direction; crashes into walls carry **-15 penalty**)

#### How to Run:
```bash
python main.py --game vault
```

#### Key Capabilities Demonstrated:
1. **Dynamic Target Switching:** Switches hypothesis goal phrase from `"the KEY"` to `"the EXIT"` the instant key possession is secured.
2. **Asymmetric Risk Management:** Incorporates specific environment penalties into $EV$ (e.g. guard proximity has higher downside than waiting).
3. **Loop Breaking via In-Context Warnings:** When hitting a hidden wall at `(3, 2)`, the bot receives a `WARNING: In the previous turn, taking action 'RIGHT' resulted in no progress`. Laya drops $P(\text{RIGHT})$, shifting maximum EV to `WAIT` or `UP`, successfully navigating around the obstacle.

#### Expected Console Output & ASCII View:
```text
  Vault Runner -- Step 30/40  Keys=0  Penalty=-7
  -----------------
  |S . . . W . . . |
  |. W . . . . W . |
  |. . P . . . . . |
  |. G W . . L . . |
  |. . . . K . . . |
  |. . . . . . W . |
  |. . . T . . . . |
  |. . . . . . . E |
  -----------------
  Legend: P=player G=guard K=key E=exit W=wall L=laser T=trap

          [!] Hidden wall discovered at (3, 2)! WARNING injected next turn.
[Step 31] Pos:(2,2) Keys:0 | Probs: ['up': 16.7%, 'down': 5.3%, 'left': 9.7%, 'right': 10.0%, 'wait': 29.6%, 'sprint': 13.0%]
          | EVs:  ['up': -2.50, 'down': -4.21, 'left': -3.54, 'right': -3.50, 'wait': -0.11, 'sprint': -5.40]
          | -> Move: 'WAIT' (EV: -0.11) | [ADAPTED: WARNING: In the previous turn, taking ac...]
```

---

### Demo 3: EV Decision Engine vs. Argmax Baseline Comparison

To demonstrate why raw probability `argmax(P)` fails in high-stakes environments, you can run Vault Runner with the EV risk-weighting disabled:

#### How to Run the Argmax Baseline:
```bash
python main.py --game vault --no-ev
```

#### What This Compares:
- **`--no-ev` (Argmax Baseline):** The bot greedily chooses $\arg\max_a P(a)$. It sprints blindly into walls (incurring crash penalties) and walks adjacent to moving guards if the forward probability is slightly higher than waiting.
- **Default (EV Engine):** The bot weights $EV = (P \times \text{Reward}) - ((1 - P) \times \text{Penalty})$. Even when `SPRINT` has a high raw confidence (e.g. 72%), the risk of a wall collision drops its EV below safe single moves or waiting:
  $$\text{EV}(\text{SPRINT}) = (0.72 \times 8.0) - (0.28 \times 15.0) = \mathbf{+1.56}$$
  $$\text{EV}(\text{DOWN}) = (0.70 \times 10.0) - (0.30 \times 5.0) = \mathbf{+5.50}$$
  The EV engine chooses `DOWN`, avoiding the catastrophic crash.

---

### Demo 4: Acceptance Tests & 20-Seed Headless Benchmark

We built an automated test harness covering all mechanics, edge cases, state compression limits, and a multi-seed statistical benchmark.

#### How to Run:
```bash
python test_vault.py
```

#### Acceptance Tests Performed:
1. `Test 1`: **Determinism + Solvability** (seeded RNG guarantees reproducible, solvable layouts)
2. `Test 2a`: **Guard Patrol** (verifies step-by-step patrol pathing and boundary bounce)
3. `Test 2b`: **Laser Hazard** (verifies continuous -5 penalty per laser traversal)
4. `Test 2c`: **Trap Floor** (verifies one-time -8 penalty and conversion to regular floor)
5. `Test 2d`: **EXIT Lock** (verifies EXIT tile remains impenetrable until Key is collected)
6. `Test 2e`: **Sprint Wall Crash** (verifies 2-tile leap and wall collision penalty)
7. `Test 3`: **State Compression** (verifies length $\le 120$ chars and zero hidden wall leakage)
8. `Test 4`: **EV vs Argmax Divergence** (verifies EV engine diverges from greedy $P$ in hazardous states)
9. `Test 5`: **Failure Adaptation** (verifies hidden wall collision triggers `WARNING` and pivots move)
10. `Test 6`: **Headless Benchmark** (executes 20 procedural seeds comparing EV vs Argmax)
11. `Test 7`: **Classic Regression Integrity** (verifies original 1D game continues running unchanged)

#### Benchmark Results (20 Randomized Seeds):
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

### Demo 5: Live Streamlit Telemetry Dashboard

An interactive real-time telemetry dashboard designed for live hackathon presentation and judges' defense.

#### How to Run:
```bash
# Terminal 1: Launch the live dashboard
streamlit run viewer.py

# Terminal 2: Run either game bot
python main.py --game vault
```

#### Dashboard Capabilities:
1. **Top-Level KPI Banners:** Total Decisions Logged, Player Coordinates `(x, y)`, Latest Action, Current EV, and Game Status (`WON` / `IN_PROGRESS` / `GAME OVER`).
2. **🗺️ Agent Position Heatmap (Vault Runner):** Visualizes the agent's spatial trajectory across the 8x8 floor grid in real-time.
3. **Calibrated Probabilities & EV Bar Graphs:** Side-by-side comparison of Laya's confidence vs calculated EV scores for every legal action.
4. **Typed Hypothesis Inspector:** Displays the exact natural-language hypothesis queries evaluated by Laya on that step.
5. **Short-Term Memory Alert Banners:** Displays active `WARNING` prompts whenever failure-adaptive memory intervenes.
6. **Chronological Audit Feed:** Complete table of historical decisions streamed directly from `decision_log.jsonl`.

---

## 📚 Master Pitch & Judge Defense Guide

For comprehensive hackathon presentation materials, architecture deep dives, and anticipated judge Q&As, consult our master briefing documents:
- 📖 [MASTER_EXPLANATION.md](file:///f:/Desktop/My%20stuff/vs%20code/ProgrammablePros/MASTER_EXPLANATION.md): Formatted markdown guide with architectural diagrams, mathematical proof walkthroughs, and presentation checklist.
- 📄 [MASTER_EXPLANATION.txt](file:///f:/Desktop/My%20stuff/vs%20code/ProgrammablePros/MASTER_EXPLANATION.txt): Synchronized plaintext reference for terminal reading and rapid offline review.