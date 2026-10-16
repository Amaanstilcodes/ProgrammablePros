"""Benchmark Tool — Genesis 2.0 Track EVA 02.

Profiles decision pipeline latency:
- State compression & feature extraction throughput
- Laya inference latency per action
- EV calculation synthesis time
"""

from __future__ import annotations

import time

from core.advanced_ev_calculator import AdvancedEVCalculator
from core.laya_backend import MockLayaBackend
from games.snake.features import SnakeFeatureExtractor
from games.snake.hypotheses import SnakeHypothesisBuilder
from games.snake.memory import SnakeMemory
from games.snake.parser import SnakeStateParser
from games.snake.subgoals import SnakeSubgoalArbiter


def benchmark_pipeline(iterations: int = 100) -> None:
    print(f"[*] Benchmarking V2 Decision Pipeline over {iterations} iterations...")

    parser = SnakeStateParser()
    features_ext = SnakeFeatureExtractor()
    subgoal_arb = SnakeSubgoalArbiter()
    hypo_builder = SnakeHypothesisBuilder()
    memory = SnakeMemory()
    backend = MockLayaBackend()
    ev_calc = AdvancedEVCalculator()

    raw_state = {
        "snake": [[5, 5], [5, 4], [5, 3]],
        "food": [10, 10],
        "direction": "right",
        "score": 60,
        "step": 1,
    }

    t0 = time.perf_counter()
    for _ in range(iterations):
        parsed = parser.parse(raw_state)
        feats = features_ext.extract_features(parsed, memory)
        subgoal = subgoal_arb.arbitrate(parsed, memory)
        hypos = hypo_builder.build_hypotheses(feats, subgoal, ["up", "down", "left", "right"])

        probs = {
            act: {
                dim: backend.predict_probability(hypos["context"], q)
                for dim, q in hypos["queries_by_action"][act].items()
            }
            for act in ["up", "down", "left", "right"]
        }

        best_act, ev_dict = ev_calc.calculate((5, 5), ["up", "down", "left", "right"], probs)

    total_time = time.perf_counter() - t0
    avg_turn_ms = (total_time / iterations) * 1000.0
    throughput = iterations / total_time

    print(f"[OK] Total time: {total_time:.3f}s")
    print(f"[OK] Average turn latency: {avg_turn_ms:.3f} ms")
    print(f"[OK] Throughput: {throughput:.1f} decisions/sec")


if __name__ == "__main__":
    benchmark_pipeline()
