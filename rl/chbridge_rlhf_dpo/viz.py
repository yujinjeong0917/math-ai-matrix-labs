"""results/chbridge.json과 logs/chbridge_samples.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/chbridge_rlhf_dpo/viz.py`

- KL 계수별 평균 곡선: 학습한 점수(r_hat)는 오르는데 참 보상(r*)은 어디서 꺾이나
- KL에 대한 참 보상: 정책경사 궤적, 정확한 최적해, 참 보상을 아는 이상적 곡선
- DPO의 걸음 수에 따른 KL
- 학습 중간중간 생성한 문장
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
BLOCKS = " .:-=+*#%@"


def spark(xs, lo, hi):
    out = []
    for x in xs:
        k = int((len(BLOCKS) - 1) * (x - lo) / (hi - lo + 1e-12))
        out.append(BLOCKS[max(0, min(len(BLOCKS) - 1, k))])
    return "".join(out)


def main():
    d = json.loads((HERE / "results" / "chbridge.json").read_text())
    print("== 정책경사(RLHF) 평균 곡선, 시드 10개 (진할수록 큼) ==")
    for b, v in d["E2_pg_summary"].items():
        c = v["mean_curve"]
        print(f"beta={b:>5}  r*   {spark([x['r_true'] for x in c], -4, 3)}")
        print(f"{'':>11}  r_hat {spark([x['r_hat'] for x in c], 0, 2)}")
        print(f"{'':>11}  KL   {spark([x['kl'] for x in c], 0, 14.3)}")

    print("\n== KL에 대한 참 보상 (정확한 최적해, 시드 평균) ==")
    lin = d["E3_exact"]["tilt_linear_rm_mean"]
    tru = d["E3_exact"]["tilt_true_reward"]
    print(f"{'beta':>6} {'KL(학습 점수)':>14} {'r*':>8} {'KL(참 보상)':>12} {'r*':>8}")
    for b in lin:
        print(f"{b:>6} {lin[b]['kl']:>14.3f} {lin[b]['r_true']:>8.3f} {tru[b]['kl']:>12.3f} {tru[b]['r_true']:>8.3f}")

    print("\n== DPO(336칸) 걸음 수에 따른 KL ==")
    for b, v in d["E4_dpo"]["full_summary"].items():
        c = v["mean_curve"]
        print(f"beta={b:>5}  KL {spark([x['kl'] for x in c], 0, 10)}   마지막 {c[-1]['kl']:.2f}, r* {c[-1]['r_true']:.2f}")

    print("\n== 생성 문장 (시드 0) ==")
    rows = [json.loads(line) for line in (HERE / "logs" / "chbridge_samples.jsonl").read_text().splitlines()]
    for row in rows:
        if row["done"]:
            info = row["info"]
            print(f"{info['config']:>20} 걸음 {info['step']:>4}  {info['text']}  r* {info['r_true']:>5}  점수 {row['r']}")


if __name__ == "__main__":
    main()
