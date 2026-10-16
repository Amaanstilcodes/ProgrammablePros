# Genesis 2.0 Track EVA 02 — Technical Writeup

**Team:** ProgrammablePros  
**Core Innovation:** Hypothesis-Driven Calibrated Probability Intelligence

---

## Executive Summary
Unlike traditional game bots relying on slow generative LLMs prone to hallucination or heavy search algorithms banned by competition rules, **ProgrammablePros** employs **Laya**, a calibrated probability model. We translate game dynamics into 4-dimensional binary scientific hypotheses ($P(\text{legal}), P(\text{progress}), P(\text{danger}), P(\text{novel})$). These calibrated probabilities are synthesized through an Expected Value (EV) decision engine, balancing reward, risk, and frontier mapping while maintaining strict compliance with official Track EVA 02 invariants.

---

## System Architecture

```
[ TopGameAI Game State ]
           │
           ▼
[ State Parser & Normalizer ] ──> [ Egocentric Feature Extractor ]
                                             │
                                             ▼
                                  [ Subgoal Arbiter ]
                                             │
                                             ▼
                                  [ Hypothesis Builder ]
                                             │
                                             ▼
                                   [ Laya Probability Model ]
                                             │
                                             ▼
                                  [ Advanced EV Calculator ]
                                             │
                                             ▼
                                  [ Compliance Safety Guard ]
                                             │
                                             ▼
                          [ Chosen Action Sent to Server ]
                                             │
                                             ▼
                   [ Dual Logger: Official Hash-Chain JSONL + Rich Trace ]
```

---

## Compliance Invariants & Audit Rigor
1. **Cryptographic SHA-256 Chains:** Every log line is hashed over canonical recursive JSON with zero whitespace and sorted keys. Genesis line anchored to `GENESIS-<session_id>`.
2. **Strict Simplex Normalization:** Probabilities guaranteed to sum within $[0.98, 1.02]$ with residual absorbed into dominant action.
3. **Safety Fallback Cap:** Anti-reversal fallback strictly capped below 10% of accepted moves.
4. **Zero State Mutation:** Raw API state dictionaries are preserved as unmodified deep copies.
