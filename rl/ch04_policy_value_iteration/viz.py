"""results/ch04.json을 터미널에 그린다(외부 패키지 없음). `uv run python rl/ch04_policy_value_iteration/viz.py`

- 6x6 격자에서 정책 반복이 반복마다 바꾼 화살표(◎ = 도착 칸)
- 동률을 무작위로 깼을 때 반복마다 바뀐 칸 수(시드 0, 처음 12번)
- 20x20 격자에서 방법별 백업 수 막대
- gamma = 1일 때 가치 반복의 ||V_k|| 증가
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def bar(x, xmax, width=40):
    return "#" * max(1, round(width * x / xmax))


def main():
    d = json.loads((HERE / "results" / "ch04.json").read_text())
    frames = d["E9"]["frames"]
    print("정책 반복 (6x6, gamma 0.95, 처음엔 모두 위)")
    for i in range(0, len(frames), 4):
        chunk = frames[i:i + 4]
        print("   ".join(f"반복 {i + j:<3}" for j in range(len(chunk))))
        for r in range(6):
            print("   ".join(f"{f[r]:<7}" for f in chunk))
        print()
    ch = d["E1"]["random"][0]["changed_first12"]
    print("동률 무작위: 반복마다 바뀐 칸 수 (시드 0)")
    for k, c in enumerate(ch, 1):
        print(f"  {k:>2} {bar(c, max(ch), 30)} {c}")
    print()
    ms = d["E3"]["methods"]
    mx = max(v["backups"] for v in ms.values())
    print("20x20, gamma 0.99: 백업 수 (정책 반복은 직접 풀이 FLOP가 따로 듦)")
    for k, v in ms.items():
        print(f"  {k:<6} {bar(v['backups'], mx)} {v['backups']:,} (반복 {v['iterations']})")
    print()
    norms = d["E2"]["1.0"]["norm"]
    print("gamma = 1, 종료 칸 없음: ||V_k||")
    for k in (1, 10, 20, 30, 40, 50):
        print(f"  k={k:>2} {bar(norms[k - 1], norms[-1])} {norms[k - 1]:.0f}")


if __name__ == "__main__":
    main()
