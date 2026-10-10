"""results/ch06.json과 logs/ch06_transitions.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch06_td_sarsa_qlearning/viz.py`

- 절벽 격자: 시드 0에서 500판 학습한 뒤 마지막 100판 동안 칸마다 지나간 횟수(Sarsa / Q-러닝 / Expected Sarsa)
- 같은 시드의 탐욕 경로(epsilon 0)
- 25판 묶음 평균 온라인 리턴 곡선(시드 100개 평균)
- 무작위 보행 RMSE(판 수별, 시드 100개 평균)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import rl_ch06_envs as E  # noqa: E402
import rl_ch06_td_np as T  # noqa: E402

SHADES = " .:-=+*#%@"


def draw_counts(env, counts):
    mx = max(counts.max(), 1)
    for r in range(env.h):
        row = ""
        for c in range(env.w):
            s = r * env.w + c
            if s in env.cliff:
                row += " X"
            elif s == env.goal:
                row += " G"
            else:
                row += " " + SHADES[min(9, int(9 * counts[s] / mx + 0.999))] if counts[s] else "  "
        print(row)


def draw_path(env, path):
    on = set(path)
    for r in range(env.h):
        row = ""
        for c in range(env.w):
            s = r * env.w + c
            row += " X" if s in env.cliff else (" G" if s == env.goal else (" o" if s in on else " ."))
        print(row)


def main():
    d = json.loads((HERE / "results" / "ch06.json").read_text())
    env = E.Cliff()
    for m, name in (("sarsa", "Sarsa"), ("q", "Q-러닝"), ("expected", "Expected Sarsa")):
        log = []
        o = T.control(env, m, 500, 0.5, 1.0, 0.1, 0, log=log)
        counts = np.zeros(env.S)
        for (ep, t, s, a, r, s2, done, fell) in log:
            if ep >= 400:
                counts[s] += 1
        print(f"== {name}: 시드 0, 401~500판에 지나간 칸 (진할수록 자주, X 절벽, G 도착) ==")
        draw_counts(env, counts)
        g = T.greedy_rollout(env, o["Q"])
        print(f"-- 탐욕 경로: 리턴 {g[0]}, {g[1]}걸음, 도착 {g[2]}")
        draw_path(env, g[3])
        print()

    print("== 25판 묶음 평균 온라인 리턴 (시드 100개, epsilon 0.1) ==")
    for key, name in (("sarsa_eps0.1", "Sarsa"), ("q_eps0.1", "Q-러닝"), ("expected_eps0.1", "Expected")):
        c = d["E3_cliff"][key]["curve_block25_mean"]
        print(f"{name:>9} " + " ".join(f"{x:7.1f}" for x in c[::2]))

    print("\n== 무작위 보행 RMSE (시드 100개 평균) ==")
    bm = d["E1_walk"]["by_method"]
    marks = ["1", "10", "25", "50", "100", "300", "1000"]
    print(" " * 10 + "".join(f"{m:>8}" for m in marks))
    for key in sorted(bm):
        print(f"{key:>10}" + "".join(f"{bm[key][m]['mean']:8.4f}" for m in marks))

    print("\n== 로그에 남은 판 ==")
    eps = {}
    for line in (HERE / "logs" / "ch06_transitions.jsonl").read_text().splitlines():
        rec = json.loads(line)
        eps.setdefault((rec["info"]["method"], rec["episode"]), []).append(rec)
    for (m, i), recs in eps.items():
        tail = recs[-1]
        state = "잘림" if tail["info"].get("truncated") else ("끝" if tail["done"] else "?")
        falls = sum(1 for r in recs if r["info"].get("fell"))
        print(f"{m} #{i}: {len(recs)}줄, 마지막 t={tail['t']}, {state}, 절벽 {falls}번, 리턴 {sum(r['r'] for r in recs):.0f}")


if __name__ == "__main__":
    main()
