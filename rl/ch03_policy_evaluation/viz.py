"""results/ch03.json을 터미널에 그린다(외부 패키지 없음). `uv run python rl/ch03_policy_evaluation/viz.py`

- 10 x 10 격자에서 스윕마다 가치가 도착 칸(오른쪽 아래)에서 퍼져 나가는 모습
- 20 x 20 오차 곡선(로그 눈금 막대)과 이론 상한 gamma^k * e0
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
SHADES = " .:-=+*#%@"


def shade_grid(grid, vmax):
    """가치 격자(행 목록)를 칸당 두 글자의 음영 문자열로 바꾼다. 0 이하인 칸은 공백.

    음영 단계 = (x/vmax)^0.5 를 SHADES 길이에 대응(제곱근은 작은 값도 눈에 띄게 하려는 선택).
    """
    rows = []
    for row in grid:
        rows.append("".join(SHADES[min(len(SHADES) - 1, int((x / vmax) ** 0.5 * (len(SHADES) - 1)))] * 2 if x > 0 else "  "
                            for x in row))
    return rows


def main():
    d = json.loads((HERE / "results" / "ch03.json").read_text())
    f = d["E6"]  # experiments.py의 e6_frames 결과(10x10 프레임, 20x20 오차 곡선)
    vmax = max(max(r) for r in f["true"])  # 모든 프레임을 정답의 최댓값 기준으로 칠해 스윕 간 비교가 가능하게 함
    for k, grid in list(f["frames"].items()) + [("정답", f["true"])]:
        print(f"스윕 {k}  (0보다 큰 칸 {f['nonzero_cells'].get(k, '-')}개)")
        for line in shade_grid(grid, vmax):
            print("  |" + line + "|")
        print()
    print("20 x 20 오차 (정답과의 최대 차이) vs 상한 gamma^k * e0")
    for k, e in f["error_curve_20"].items():
        b = f["error_bound_20"][k]  # 상한 gamma^k * ||V_0 - V^pi|| (V_0 = 0 이므로 e0 = ||V^pi||)
        # 막대 길이 = 20 + 3 log10(e): 오차가 10배 줄 때마다 3칸 짧아진다(로그 눈금)
        bar = "#" * max(1, int(20 + 3 * math.log10(e)))
        print(f"  k={k:>5}  {e:9.2e}  상한 {b:9.2e}  {bar}")
    log = [json.loads(x) for x in (HERE / "logs" / "ch03_rollouts.jsonl").read_text().splitlines()]
    print(f"\n로그 {len(log)}줄, 첫 줄: {log[0]}")


if __name__ == "__main__":
    main()
