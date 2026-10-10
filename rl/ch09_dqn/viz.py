"""results/ch09.json과 logs/ch09_transitions.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch09_dqn/viz.py`

- 최댓값의 치우침: K별 단일 추정, Double, 수치적분
- 조건별 시작 상태 점수 max Q(s0) 곡선(시드별)과 실제 할인 리턴
- 조건별 마지막 100판 평균 리턴
- 학습 뒤 탐욕 정책 한 판의 막대 각도·수레 위치 궤적
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def bar(x, xmax, width=40):
    x = max(0.0, float(x))
    return "#" * min(width, max(1 if x > 0 else 0, round(width * x / xmax)))


def main():
    d = json.loads((HERE / "results" / "ch09.json").read_text())
    print("== 참값이 모두 0인 행동 K개, 잡음 N(0,1): 고른 값의 평균 ==")
    for K, r in d["E1_max_bias"]["equal_true_values_0"].items():
        print(f"K={K:>2} 단일 {r['single']:6.3f} (적분 {r['integral']:6.3f}) {bar(r['single'], 2.5)}")
        print(f"     Double {r['double']:6.3f}")

    print("\n== 시작 상태 점수 max Q(s0), 2,500걸음마다 (참값은 100을 넘을 수 없음) ==")
    runs = d["E4_conditions"]["runs"]
    for name, rs in runs.items():
        print(f"-- {name}")
        for sd, r in zip(d["E4_conditions"]["seeds"], rs):
            q = r["eval_q0"]
            print(f"  시드 {sd}: " + " ".join(f"{x:5.0f}" for x in q[::2]))
        g = rs[0]["eval_g0"]
        print("  실제 리턴(시드 0): " + " ".join(f"{x:5.0f}" for x in g[::2]))

    print("\n== 마지막 100판 평균 리턴 (시드 5개) ==")
    for name, s in d["E4_conditions"]["summary"].items():
        print(f"{name:22s} {s['last100_mean']:6.1f} {bar(s['last100_mean'], 500)}  max|Q| 중앙값 {s['max_abs_q_median']:.0f}")

    print("\n== 학습 뒤 탐욕 정책 한 판 (각도는 도, 위치는 m) ==")
    rows = [json.loads(x) for x in (HERE / "logs" / "ch09_transitions.jsonl").read_text().splitlines()]
    ev = [r for r in rows if r["episode"] == "eval"]
    for r in ev[::25]:
        x, th = r["s"][0], r["s"][2] * 180 / 3.141592653589793
        pos = int(round((x + 2.4) / 4.8 * 40))
        line = [" "] * 41
        line[20] = "|"
        line[max(0, min(40, pos))] = "o"
        print(f"t={r['t']:3d} 각도 {th:6.2f} 위치 {x:6.3f} [{''.join(line)}] Q={r['info']['q']}")
    print(f"판 길이 {len(ev)}걸음")


if __name__ == "__main__":
    main()
