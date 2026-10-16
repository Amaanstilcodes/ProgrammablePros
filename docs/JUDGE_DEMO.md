# Judge Demo & Verification Guide

**Genesis 2.0 Track EVA 02**  
**Team:** ProgrammablePros  
**Architecture:** Laya Calibrated Probability Engine + Multi-Objective EV + Strict Cryptographic Hashing

---

## 1. Quick Verification (30 Seconds)

Run our automated acceptance test suite covering all 23 competition invariants:
```bash
python -m pytest tests/ -v
```

---

## 2. Execute a Compliant Session (Snake)

Run a clean 10-step mock session generating canonical hash-chained logs:
```bash
python orchestrator_compliance.py --game snake --mock --max-steps 10
```

Verify the generated log file independently using our CLI auditor:
```bash
python tools/verify_log.py logs/_mock/<SESSION_ID>/decision_log_<SESSION_ID>.jsonl
```

---

## 3. Launch the Live Telemetry Dashboard

Launch the live Streamlit viewer to observe decisions, probability simplex distributions, and ASCII board views:
```bash
streamlit run viewer_compliance.py
```

---

## 4. Run Legacy Benchmark (Proof of Multi-Objective EV Dominance)

Run our 20-seed headless benchmark comparing our risk-aware EV engine against standard argmax:
```bash
python test_vault.py
```
*(Demonstrates 90% win rate and $5\times$ penalty reduction).*
