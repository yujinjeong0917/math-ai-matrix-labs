"""results/ch08.json과 logs/ch08_offpolicy.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch08_function_approx/viz.py`

- 가중치 궤적: gamma 0.99와 0.9에서 체크포인트마다 w0, w6, V(6), 값 오차(부호를 붙인 log10)
- alpha x gamma 지도: 이론(스펙트럼 반경)과 실측 판정
- 오프폴리시 궤적 로그: 시드 0 처음 600걸음, 50걸음마다 가중치 노름 막대
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent


def slog(x):
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    return f"{'+' if x > 0 else '-'}1e{math.log10(abs(x)):.1f}"


def weight_tracks(res):
    for key, run in res["E1_fail"].items():
        print(f"\n[{key}] alpha {run['alpha']}, 스윕별 (부호를 붙인 크기)")
        print(f"{'스윕':>6} {'w0':>9} {'w6':>9} {'V(6)':>9} {'값 오차':>9}")
        for r in run["checkpoints"]:
            print(f"{r['sweep']:>6} {slog(r['w0']):>9} {slog(r['w6']):>9} {slog(r['V6']):>9} {slog(r['value_max']):>9}")
        print(f"  부호가 바뀐 횟수: w0 {run['sign_changes']['w0']}번, w6 {run['sign_changes']['w6']}번")


def grid(res):
    print("\nalpha x gamma (10,000 스윕, 처음 값 오차 12). 이론 / 실측")
    rows = res["E4_grid"]["rows"]
    gammas = sorted({r["gamma"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})
    print("gamma  " + "".join(f"{a:>22}" for a in alphas))
    for g in gammas:
        cells = []
        for a in alphas:
            r = next(x for x in rows if x["gamma"] == g and x["alpha"] == a)
            cells.append(f"{r['theory']}/{r['measured']} ({r['spectral_radius']})")
        print(f"{g:<6} " + "".join(f"{c:>22}" for c in cells))


def offpolicy_log():
    rows = [json.loads(line) for line in (HERE / "logs" / "ch08_offpolicy.jsonl").open()]
    print("\n오프폴리시 궤적 (시드 0, gamma 0.99, alpha 0.01). 50걸음마다 가중치 노름, # 하나 = 5")
    for r in rows[::50]:
        n = r["info"]["w_norm"]
        print(f"t={r['t']:>4} s={r['s']} rho={r['info']['rho']:.0f} |w|={n:8.2f} " + "#" * min(60, int(n / 5)))


if __name__ == "__main__":
    res = json.loads((HERE / "results" / "ch08.json").read_text())
    weight_tracks(res)
    grid(res)
    offpolicy_log()
