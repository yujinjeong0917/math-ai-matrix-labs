"""results/ch07.json과 logs/ch07_traces.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch07_nstep_td_lambda/viz.py`

- 흔적 히트맵: 시드 0 첫 판, λ 0.8. 줄 = 걸음(시간), 칸 = 상태 1~13. 누적 흔적과 대체 흔적을 나란히.
- n x alpha 표: 10판 뒤 RMSE(시드 0~99 평균)를 글자 농도로. 가장 작은 칸에 O
- λ x alpha 표: 온라인 누적 흔적. 30판 안에 발산한 시드가 있으면 X
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SHADES = " .:-=+*#%@"


def shade(x, lo, hi):
    if x is None:
        return "X"
    k = int(9 * (x - lo) / (hi - lo) + 0.5) if hi > lo else 0
    return SHADES[max(0, min(9, k))]


def trace_heatmap():
    rows = [json.loads(line) for line in (HERE / "logs" / "ch07_traces.jsonl").open()]
    print("흔적 히트맵 (시드 0 첫 판, λ 0.8). 왼쪽: 누적, 오른쪽: 대체. 농도는 0 ~ 2.5")
    print("  t  s   " + "".join(f"{i % 10}" for i in range(1, 14)) + "   " + "".join(f"{i % 10}" for i in range(1, 14)))
    for r in rows:
        a = "".join(shade(min(x, 2.5), 0, 2.5) for x in r["info"]["trace_accumulating"])
        b = "".join(shade(min(x, 2.5), 0, 2.5) for x in r["info"]["trace_replacing"])
        print(f"{r['t']:>3} {r['s']:>2}   {a}   {b}")


def grid_tables(res):
    e1 = res["E1_nstep"]
    alphas = e1["alphas"]
    print("\n10판 뒤 RMSE, n x alpha (시드 0~99 평균). 진할수록 오차가 크다, O = 그 n에서 가장 작은 alpha")
    print("   n   " + " ".join(f"{a:<4}"[:4] for a in alphas))
    vals = [v for g in e1["grid"].values() for v in g["after10_mean"]]
    lo, hi = min(vals), max(vals)
    for n, g in e1["grid"].items():
        best = e1["best"][n]["after10"]["alpha"]
        cells = []
        for a, v in zip(alphas, g["after10_mean"]):
            cells.append(("O" if a == best else shade(v, lo, hi)).ljust(4))
        print(f"{n:>4}   " + " ".join(cells))
    vo = res["E2_lambda"]["variants"]["online_acc"]
    print("\n같은 표, 온라인 누적 흔적 TD(λ). X = 30판 안에 발산한 시드가 있음")
    for lam, g in vo["grid"].items():
        cells = []
        for v, d in zip(g["after10_mean"], g["diverged_seeds"]):
            cells.append(("X" if d else shade(v, lo, hi)).ljust(4))
        print(f"{lam:>4}   " + " ".join(cells))


def main():
    res = json.loads((HERE / "results" / "ch07.json").read_text())
    trace_heatmap()
    grid_tables(res)
    p = res["E3_compare"]["paired"]
    print(f"\n새 시드 100개: 가장 좋은 n = {p['best_n']}, 가장 좋은 λ(온라인 누적) = {p['best_lam_online_acc']}, "
          f"n스텝이 더 작았던 시드 {p['seeds_best_n_smaller_than_best_lam']}개")


if __name__ == "__main__":
    main()
