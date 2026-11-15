"""results/ch01.json과 logs/ch01_two_arm.jsonl을 터미널에 막대로 그린다(외부 패키지 없음).

`uv run python rl/ch01_bandit/viz.py`
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


# x를 scale로 나눈 비율만큼 '#'을 채운다. scale 초과는 한 줄 폭(width)에서 자른다
def bar(x, scale, width=40, ch="#"):
    return ch * int(round(min(x / scale, 1.0) * width))


def main():
    d = json.loads((HERE / "results" / "ch01.json").read_text())
    e2 = d["E2"]
    # E2: 10팔 p = 0.10~0.55, 한 규칙이 손잡이를 평균 몇 번 당겼는지(막대 척도 1000 = 전체 횟수)
    print("손잡이별 평균 당긴 횟수 (10팔, 1,000번, 2,000 반복 평균)")
    for name, row in e2["by_policy"].items():
        print(f"\n[{name}]")
        for p, c in zip(e2["p"], row["mean_pulls_per_arm"]):
            print(f"  p={p:.2f} {c:7.1f} {bar(c, 1000)}")
    # 누적 후회 = sum_j Delta_j T_j(t). 체크포인트 t별 반복 평균과 5%, 95% 분위수
    print("\n누적 후회: 평균 [5%, 95%] (팬 차트를 숫자로)")
    for name, row in e2["by_policy"].items():
        cells = [f"{k}: {row['regret_mean_at'][k]:.1f} [{row['regret_p5_at'][k]:.1f}, {row['regret_p95_at'][k]:.1f}]"
                 for k in row["regret_mean_at"]]
        print(f"  {name:16s} " + " | ".join(cells))
    # Lai-Robbins 곡선 = C ln n (점근 하한의 앞부분 상수만 쓴 곡선)
    print(f"  Lai-Robbins 곡선  " + " | ".join(f"{k}: {v:.1f}" for k, v in e2["lai_robbins_at"].items()))
    # E1에서 저장한 2팔 욕심쟁이 로그(좋은 손잡이 첫 결과 0, 나쁜 손잡이 첫 결과 1인 시드)
    log = [json.loads(line) for line in (HERE / "logs" / "ch01_two_arm.jsonl").read_text().splitlines()]
    print(f"\n2팔 욕심쟁이 로그 {len(log)}줄 (시드 {log[0]['seed']}):")
    for rec in log:
        print(f"  t={rec['t']:2d} 손잡이 {rec['a']} 보상 {rec['r']:.0f}")


if __name__ == "__main__":
    main()
