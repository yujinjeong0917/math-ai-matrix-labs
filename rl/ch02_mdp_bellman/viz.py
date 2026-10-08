"""results/ch02.json과 logs/ch02_rollouts.jsonl을 터미널에 격자로 그린다(외부 패키지 없음).

`uv run python rl/ch02_mdp_bellman/viz.py`
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def show_grid(title, grid, fmt="{:>6}"):
    print(title)
    for row in grid:
        print(" ".join(fmt.format(x if isinstance(x, str) else f"{x:.2f}") for x in row))
    print()


def main():
    d = json.loads((HERE / "results" / "ch02.json").read_text())
    show_grid("V* (slip 0, gamma 0.9)", d["E0"]["V_star_grid"])
    for p in ("myopic", "optimal"):
        show_grid(f"방문 비율 ({p}, 시작 (0,0))", d["E1"][p]["visit_frac_grid"])
    for slip, row in d["E3"]["by_slip"].items():
        show_grid(f"최적 정책 화살표 (slip {slip})", row["optimal_arrows"], fmt="{:>2}")
    log = [json.loads(line) for line in (HERE / "logs" / "ch02_rollouts.jsonl").read_text().splitlines()]
    print(f"로그 {len(log)}줄, 첫 줄: {log[0]}")


if __name__ == "__main__":
    main()
