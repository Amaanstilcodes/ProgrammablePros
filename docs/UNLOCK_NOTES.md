# Stage Transition & Unlock Notes

**Genesis 2.0 Track EVA 02**  
**Team:** ProgrammablePros

---

## Stage Progression Gate (Rule R8)

> *"Never run Stage 2 (2048) before a verified Snake run >= 600 plus an explicit organizer unlock confirmation."*

### Unlock Criteria
1. **Stage 1 (Snake):** Achieve a score of $\ge 600$ points (eating at least 10 foods).
2. **Hash Chain Verification:** The session's `decision_log_<sid>.jsonl` must pass 100% cryptographic SHA-256 verification via `OfficialLogger.verify_log_file()`.
3. **Organizer Confirmation:** Confirmation token stored in `submission/verified/stage1_unlock.json` formatted as:
   ```json
   {
     "verified_session_id": "<SESSION_ID>",
     "final_score": 660,
     "organizer_token": "EVA02-STAGE2-AUTHORIZED",
     "timestamp": "2026-11-07T10:00:00Z"
   }
   ```

### Enforcement Mechanism
The `core.stage_gate.StageGate` class automatically checks for this token or scans `logs/snake/` before allowing `orchestrator_compliance.py --game 2048` to proceed. If unverified, it raises `StageLockedError`.
