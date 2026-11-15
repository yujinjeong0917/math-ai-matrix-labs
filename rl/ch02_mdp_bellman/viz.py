"""results/ch02.json과 logs/ch02_rollouts.jsonl을 터미널에 격자로 그린다(외부 패키지 없음).

`uv run python rl/ch02_mdp_bellman/viz.py`
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


# grid: 문자열이면 그대로, 숫자면 소수 둘째 자리로 찍는다(화살표, 'T' 같은 표시용 칸 포함)
def show_grid(title, grid, fmt="{:>6}"):
    print(title)
    for row in grid:
        print(" ".join(fmt.format(x if isinstance(x, str) else f"{x:.2f}") for x in row))
    print()


def main():
    d = json.loads((HERE / "results" / "ch02.json").read_text())
    # E0: 최적가치 V*. 최적 경로 위 칸은 +10 칸에서 k걸음 전이면 10*0.9^(k-1)
    show_grid("V* (slip 0, gamma 0.9)", d["E0"]["V_star_grid"])
    # E1: 한 에피소드에서 각 칸을 한 번이라도 지난 비율(방문 비율). 근시 정책(gamma=0 효과)과 최적 정책 비교
    for p in ("myopic", "optimal"):
        show_grid(f"방문 비율 ({p}, 시작 (0,0))", d["E1"][p]["visit_frac_grid"])
    # E3: slip별 최적 정책의 행동(Q*의 argmax)을 화살표로, 종료 칸은 T
    for slip, row in d["E3"]["by_slip"].items():
        show_grid(f"최적 정책 화살표 (slip {slip})", row["optimal_arrows"], fmt="{:>2}")
    log = [json.loads(line) for line in (HERE / "logs" / "ch02_rollouts.jsonl").read_text().splitlines()]
    print(f"로그 {len(log)}줄, 첫 줄: {log[0]}")


if __name__ == "__main__":
    main()
