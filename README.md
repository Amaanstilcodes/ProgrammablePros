# 🤖 ProgrammablePros: Laya-Powered Autonomous Game Bot

**Hackathon:** Genesis 2.0 | **Track:** EVA 02  
**Problem Statement:** Build the decision layer for hosted online games using the Laya calibrated probability model.

## 📖 Project Overview

Most "AI plays a game" projects rely on large, slow, and expensive generative LLMs that hallucinate actions. This project takes a different approach. We built an autonomous bot that uses **Laya**, a small, open-weight calibrated probability decision model.

Instead of asking an AI "What should I do?", our bot translates the game state into typed, binary hypothesis questions (e.g., "What is the calibrated probability that executing 'RIGHT' progresses toward the goal?"). Laya returns fast, mathematically sound probabilities, which our bot evaluates using an Expected Value (EV) risk-reward calculator to make optimal, auditable decisions.

## 🎮 Competition Games

### Stage 1: Snake
- **Site:** `https://topgameai.com/arena/snake`
- **Grid:** 20x20
- **Objective:** Eat 15 foods to win
- **Rules:** Cannot reverse direction in a single move (illegal → rejected); wall/body collision ends game; max 300 accepted moves
- **Scoring:** `foods_eaten * 60` (+100 if win), capped at 1000
- **Unlock Stage 2:** One verified run with ≥ 600 points (10+ foods)

### Stage 2: 2048
- **Site:** `https://topgameai.com/arena/2048`
- **Grid:** 4x4
- **Objective:** Merge toward 512, 1024, 2048 tile
- **Rules:** Merge equal adjacent tiles once per move; spawn a 2 (90%) or 4 (10%) in a random empty cell; move that doesn't change the board is rejected; max 2000 accepted moves
- **Scoring:** `Points = min(1000, max(0, (log2(best_tile) - 6) * 143 + floor(score/1000)))`
- **Unlock Stage 3:** One verified run reaching a 512 tile
- **Judging:** Median of your three best verified runs

## 🚀 Our Unique Approach

The problem statement notes: *"Teams using the same model will score very differently, and the games will punish weak understanding."* To win, we engineered advanced strategies directly into our decision layer:

### 🧹 State Compression & Feature Engineering
Small models get confused by noisy, raw JSON. We built compressors that strip away UI noise and feed Laya high-signal variables:
- **Snake (`SnakeCompressor`):** head position, direction, body length, food coordinates, Manhattan distance to food, wall distances, body adjacency bits, foods eaten, steps remaining.
- **2048 (`Board2048Compressor`):** max tile, log2 value, corner/edge position, empty cell count, board smoothness, monotonicity, score, steps remaining.

### 🎯 Hypothesis-Driven "Typed Questions"
We frame prompts as strict, testable hypotheses rather than open-ended queries, forcing Laya to do what it does best: binary/multi-class probability calibration.
- **Progress Hypothesis (Q1):** Probability that moving in direction `a` progresses toward the active tactical subgoal.
- **Danger Hypothesis (Q2):** Probability that moving in direction `a` causes an immediate collision or paralyzes the board.

### ⚖️ Expected Value (EV) Decision Making
We don't blindly pick the highest probability. We calculate:
$$\text{EV}(a) = \big(P_{\text{progress}}(a) \times W_{\text{progress}}\big) + \big(P_{\text{structure}}(a) \times W_{\text{structure}}\big) - \big(P_{\text{danger}}(a) \times W_{\text{danger}}\big) - \text{penalty}_{\text{repeat}}(a) - \text{penalty}_{\text{illegal}}(a)$$
This mathematical risk-reward weighting guarantees optimal play and avoids catastrophic high-probability suicide moves.

### 🧠 Failure-Adaptive Prompting (Short-Term Memory)
If a move results in no progress or is rejected, the bot logs this outcome. On the next turn, it dynamically appends a `WARNING` to Laya's prompt for that specific action, forcing the model to adjust its probability downward and triggering an immediate course correction.

### 🔐 Cryptographic Audit Chain
Every move is bound into an immutable SHA-256 hash-chain JSONL log ($H_n = \text{SHA256}(H_{n-1} \mid \text{canonical\_json})$), guaranteed 4-way probability normalization summing to $1.000000 \in [0.98, 1.02]$, and capped at $\le 10\%$ safety fallback.

---

## 🏗️ Architecture & Tech Stack

| Technology | Role in Project | Description |
|---|---|---|
| **Python 3.10+** | Core Runtime & Logic | Clean type hints, dataclasses, standard library |
| **Laya Inference** | Decision Intelligence | Calibrated probability likelihood scoring |
| **Snake Brain** | Stage 1 Decision Engine | Compressor, SubgoalManager, HypothesisBuilder, EVCalculator |
| **2048 Brain** | Stage 2 Decision Engine | Compressor, SubgoalManager, HypothesisBuilder, EVCalculator |
| **Compliance Layer** | PDF Compliance Bridge | Probability normalization, fallback checks, decision block builder |
| **Official Logger** | Cryptographic Audit Trail | Immutable SHA-256 hash-chained JSONL logging |
| **Local Game Engines** | Local Test Harness | In-process simulation of Snake & 2048 mimicking hosted API response schemas |
| **Streamlit Viewer** | Live Telemetry Dashboard | Real-time monitoring of boards, confidence, EV, and audit feeds |

---

## 📁 Project Structure

```text
ProgrammablePros/
├── snake_brain.py           # Stage 1: Snake compressor, subgoals, hypotheses, and EV
├── game2048_brain.py        # Stage 2: 2048 compressor, subgoals, hypotheses, and EV
├── compliance_layer.py      # Genesis 2.0 PDF compliance bridge & probability normalizer
├── official_logger.py       # SHA-256 cryptographic hash-chained JSONL logger
├── laya_agent.py            # Laya model interface & failure-adaptive memory
├── calibration_engine.py    # Platt scaling logistic calibration & Brier scoring
├── run_local_games.py       # Unified CLI runner for local test harness
├── viewer_local.py          # Streamlit live telemetry viewer for Snake & 2048
├── local_game_clients.py    # Local client wrappers mimicking TopGameAI API
├── snake_game.py            # Local test harness: Snake simulation engine
├── game2048.py              # Local test harness: 2048 simulation engine
├── game_clients.py          # Hosted TopGameAI API client (live arena integration)
├── orchestrator_compliance.py # Hosted arena orchestration & stage gate runner
├── MASTER_EXPLANATION.md    # Master pitch briefing & AI model technical knowledge base
├── MASTER_EXPLANATION.txt   # Synchronized plain-text technical briefing
└── requirements.txt         # Minimal production dependencies
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Local Snake Benchmark
```bash
python run_local_games.py --game snake --seed 42 --use-mock-laya
```
*Expected: 15/15 foods eaten, score 1000, 0% fallback, hash chain VALID.*

### 3. Run Local 2048 Benchmark
```bash
python run_local_games.py --game 2048 --seed 42 --use-mock-laya
```
*Expected: 300+ steps, tile 256+, 0% fallback, hash chain VALID.*

### 4. Launch Live Decision Viewer
```bash
streamlit run viewer_local.py
```
Open `http://localhost:8501` to view live board grids, probability bar charts, EV telemetry, and the cryptographic audit chain.