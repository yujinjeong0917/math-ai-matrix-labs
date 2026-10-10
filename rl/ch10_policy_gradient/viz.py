"""results/ch10.json과 logs/ch10_episodes.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch10_policy_gradient/viz.py`

- 짧은 복도의 J(p) 곡선
- 같은 theta에서 뽑은 기울기 추정 1,000개의 히스토그램(베이스라인별, 시드 0)
- cart-pole 학습 곡선 팬(20판 이동평균의 시드 10개 최소, 중앙값, 최대)
- 로그에 남은 판
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import rl_ch10_corridor as cor  # noqa: E402
import rl_ch10_pg_np as m  # noqa: E402


def bar(x, xmax, width=40):
    return "#" * max(1 if x > 0 else 0, round(width * x / xmax))


def main():
    d = json.loads((HERE / "results" / "ch10.json").read_text())
    print("== 짧은 복도 J(p) = -(도착까지 걸음 수의 기댓값) ==")
    for p in np.linspace(0.05, 0.95, 19):
        J = cor.exact_v(p)[0]
        print(f"p={p:4.2f} {J:8.2f} " + bar(100 + J, 100))
    print(f"가장 좋은 p = {d['E0_hand']['best_p']}, J = {d['E0_hand']['best_J']}")

    print("\n== theta=0에서 기울기 추정 1,000개 (시드 0) ==")
    rng = np.random.default_rng(0)
    eps = [m.corridor_episode(0.0, rng) for _ in range(1000)]
    edges = np.arange(-60, 61, 10)
    for b in ("none", "const", "value"):
        g = np.array([m.corridor_grad_estimate(e, 0.0, b) for e in eps])
        h, _ = np.histogram(np.clip(g, -59.9, 59.9), edges)
        print(f"-- {b}: 평균 {g.mean():.2f}, 표준편차 {g.std(ddof=1):.2f} (정확한 기울기 2.0)")
        for lo, c in zip(edges[:-1], h):
            print(f"  [{lo:4d},{lo + 10:4d}) {c:4d} " + bar(c, 1000, 50))

    print("\n== cart-pole 20판 이동평균 (시드 10개: 최소 / 중앙값 / 최대) ==")
    for meth, r in d["E4_cartpole"]["report"].items():
        print(f"-- {meth} ({r['chosen']})")
        for ep, v in r["fan_mov20"].items():
            print(f"  {int(ep):5d}판 {v['min']:6.1f} {v['median']:6.1f} {v['max']:6.1f} " + bar(v["median"], 500))

    print("\n== 로그 ==")
    for line in (HERE / "logs" / "ch10_episodes.jsonl").read_text().splitlines()[:12]:
        print(" ", line)


if __name__ == "__main__":
    main()
