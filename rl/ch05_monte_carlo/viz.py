"""results/ch05.json과 logs/ch05_episodes.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch05_monte_carlo/viz.py`

- 20x20 격자 균등 무작위 정책의 판 길이 히스토그램(시드 10개 x 200판, 상한 10,000걸음)
- 상태 하나짜리 문제에서 일반 IS와 가중 IS 추정값의 시드 100개 분포(판 수별 중앙값, 최소, 최대)
- 로그에 남은 판들의 경로
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import rl_ch05_grid as g  # noqa: E402
import rl_ch05_mc_np as m  # noqa: E402


def bar(x, xmax, width=40):
    return "#" * max(1 if x > 0 else 0, round(width * x / xmax))


def main():
    d = json.loads((HERE / "results" / "ch05.json").read_text())
    print("== 20x20 균등 무작위 정책: 판 길이 (실험과 같은 시드로 다시 걷기) ==")
    P, R, term = g.corner_grid(20)
    L = np.concatenate([m.random_walk_lengths(200, np.random.default_rng(sd), P.argmax(axis=2), term, 0, 10_000)[0]
                        for sd in range(10)])
    edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000, 10_001]
    counts = np.histogram(L, bins=edges)[0]
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        label = "10000 (잘림)" if lo == 10_000 else f"{lo:>5}~{hi - 1:<5}"
        print(f"{label:>13} {c:>4} {bar(c, counts.max())}")
    print(f"평균 {d['E1_waiting']['e_uniform_20x20']['mean']}, 정확한 기대값 {d['E1_waiting']['e_uniform_20x20']['expected_steps_exact']}")

    print("\n== 상태 하나짜리 문제: 시드 100개의 추정값 (참값 = p / (1 - p)) ==")
    for p, c in d["E4_importance"]["cases"].items():
        print(f"p = {p}, 참값 {c['truth']}, 분산 {'무한' if c['variance_infinite'] else c['variance_exact']}")
        for N, r in c["by_N"].items():
            print(f"  N={N:>6}  일반 중앙값 {r['ordinary_median']:>8.3f} [{r['ordinary_min']:.2f}, {r['ordinary_max']:.2f}]"
                  f"  가중 중앙값 {r['weighted_median']:>8.3f}  10% 안: 일반 {r['ordinary_within10']:>3} / 가중 {r['weighted_within10']:>3}")

    print("\n== 로그에 남은 판 ==")
    eps = {}
    for line in (HERE / "logs" / "ch05_episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        eps.setdefault((rec["info"]["env"], rec["episode"]), []).append(rec)
    for (env, i), recs in eps.items():
        tail = recs[-1]
        state = "잘림" if tail["info"].get("truncated") else ("끝" if tail["done"] else "?")
        print(f"{env} #{i}: 기록 {len(recs)}줄, 마지막 t={tail['t']}, {state}, 상태 {[r['s'] for r in recs][:12]}")


if __name__ == "__main__":
    main()
