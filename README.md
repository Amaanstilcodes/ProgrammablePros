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

The problem statement notes: *"Teams using the same model will score very differently, and the games will punish weak understanding."* To win, we engineered four advanced strategies directly into our decision layer:

1. **🧹 State Compression & Feature Engineering:** Small models get confused by noisy, raw JSON. We built a compressor that strips away UI noise and feeds Laya only high-signal variables (e.g., `player_pos`, `goal_pos`, `distance_to_goal`, `steps_remaining`).
2. **🎯 Hypothesis-Driven "Typed Questions":** We frame prompts as strict, testable hypotheses rather than open-ended queries, forcing Laya to do what it does best: binary/multi-class probability calibration.
3. **⚖️ Expected Value (EV) Decision Making:** We don't blindly pick the highest probability. We calculate $EV = (P_{success} \times Reward) - (P_{failure} \times Penalty)$ to weigh the risk vs. reward of every move, preventing catastrophic failures.
4. **🧠 Failure-Adaptive Prompting (Short-Term Memory):** If a move results in no progress, the bot logs this outcome. On the next turn, it dynamically appends a `WARNING` to Laya's prompt for that specific action, forcing the model to adjust its probability downward and triggering an immediate course correction.

---

## 🏗️ Architecture & Tech Stack

| Technology | Role in Project | Engineering Justification |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Runtime & Logic | Native support for modern typing, async capabilities, and ML SDKs. |
| **Laya Inference** | Decision Intelligence | Evaluates typed hypotheses and returns calibrated probabilities ($0.0 \le P \le 1.0$), preventing generative hallucinations. |
| **Requests / REST** | Game API Communication | Clean HTTP abstraction separating the game server from the decision bot for seamless mock-to-production swapping. |
| **JSON Lines (JSONL)** | Audit Trail & Logging | Append-only streaming format that eliminates file locks, allowing concurrent reading by the dashboard while the bot writes. |
| **Streamlit** | Live Telemetry Dashboard | Zero-friction, reactive web UI for real-time monitoring of bot confidence, EV scores, and audit records. |

### Project Structure
```text
ProgrammablePros/
├── main.py             # Core game loop, EV calculation, and failure-adaptive memory tracking
├── laya_agent.py       # State compression, hypothesis formulation, and Laya API integration
├── mock_game.py        # Simulated 1D grid game environment (swapped for real API on hackathon day)
├── logger.py           # Append-only JSONL decision logging utility
├── viewer.py           # Streamlit live decision dashboard with real-time polling
├── requirements.txt    # Python dependencies
└── README.md           # This file