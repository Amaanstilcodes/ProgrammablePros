# ProgrammablePros — AI Game Bot with Laya Decision Calibration

Autonomous game-playing AI bot architecture built for hosted online games. The bot reads live game state via API, formulates typed hypothesis questions for the **Laya** calibrated probability model to evaluate, executes the highest-confidence action, logs every decision cycle to `decision_log.jsonl`, and streams live telemetry to a Streamlit dashboard.

---

## Architecture Overview

1. **`mock_game.py`**: Simulates the hosted game environment (`get_state()`, `send_action()`), tracking step count, player position, goal, and win status.
2. **`laya_agent.py`**: Interacts with the Laya decision model by translating state and candidate actions into typed probability questions and returning calibrated probabilities. Includes mock simulator and a marked `TODO` section for the real SDK.
3. **`logger.py`**: Thread-safe JSONL logger writing `timestamp`, `game_state`, `questions_asked`, `probabilities_returned`, and `chosen_action`.
4. **`main.py`**: Core game loop coordinating state perception, Laya probability scoring, argmax action execution, and logging.
5. **`viewer.py`**: Real-time Streamlit telemetry viewer with auto-refresh, metric KPI cards, probability progress bars, and historical decision breakdown.

---

## How to Run

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Start the Live Streamlit Viewer (Terminal 1)
```powershell
streamlit run viewer.py
```
*Opens at `http://localhost:8501`. Initially waits for decisions to start streaming.*

### Step 3: Run the AI Game Bot Loop (Terminal 2)
```powershell
python main.py
```
*Executes the autonomous loop, prints calibrated probabilities to the console, and streams state changes to the live viewer until goal is reached.*