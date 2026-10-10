"""results/ch11.json과 logs/ch11_updates.jsonl을 터미널에 그린다(외부 패키지 없음).
`uv run python rl/ch11_trpo_ppo/viz.py`

- 설정별 평균 학습 곡선(시드 10개)과 가장 나빴던 시드의 곡선
- 갱신별 KL과 리턴을 나란히: KL이 튄 직후 리턴이 무너지는 구간
- 시드 10개의 최종 리턴 점그림
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
BLOCKS = " .:-=+*#%@"


def spark(xs, top=200.0):
    return "".join(BLOCKS[min(len(BLOCKS) - 1, int(len(BLOCKS) * x / (top + 1e-9)))] for x in xs)


def main():
    d = json.loads((HERE / "results" / "ch11.json").read_text())
    print("== 평균 학습 곡선 (갱신 60번, 글자가 진할수록 200걸음에 가까움) ==")
    for k, v in d["curves_mean_by_update"].items():
        print(f"{k:>22} 평균 {spark(v)}")
        print(f"{'':>22} 최저 {spark(d['curves_seed_min_by_update'][k])}")

    print("\n== 갱신별 리턴과 그 갱신의 KL (로그에서) ==")
    rows = [json.loads(line) for line in (HERE / "logs" / "ch11_updates.jsonl").read_text().splitlines()]
    for cfg in ("pg_lr1000", "ppo_noclip_adam0.2", "ppo_eps0.2_adam0.2"):
        for seed in (0, 1):
            r = [x for x in rows if x["config"] == cfg and x["seed"] == seed]
            kl = [x["kl_after_update"] if x["kl_after_update"] is not None else 99.0 for x in r]
            print(f"{cfg} 시드 {seed}")
            print("  리턴 " + spark([x["return"] for x in r]))
            print("  KL   " + "".join("#" if k > 1 else "+" if k > 0.1 else "." if k > 0.01 else " " for k in kl)
                  + "   (# > 1, + > 0.1, . > 0.01)")

    print("\n== 시드 10개의 최종 리턴 (0 ~ 200) ==")
    for k, v in {**d["E1_E2_E3_summary"], **d["E3_adam_lr_sweep"]}.items():
        line = [" "] * 41
        for f in v["final_per_seed"]:
            line[min(40, int(f / 5))] = "o"
        print(f"{k:>22} |{''.join(line)}| 평균 {v['final_mean']:.1f}, 95% 구간 {v['final_ci95']}")


if __name__ == "__main__":
    main()
