"""AI 도구 실전 7장 실험. `uv run python tools/ch07_image_video/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 한 번 뽑기 2,000원, 마음에 들 확률 p = 1/4(가정), 최대 k번
E1 과금 단위: 토큰으로 매기는 모델도 결국 초·장으로 환산된다(Omni Flash, Nano Banana, GPT Image 2.5)
E2 보고서용 15초 영상을 모델마다 어떻게 채우나(한 번에 / 확장 / 이어 붙이기)와 한 번 뽑기 값
E3 15초 영상 1개를 얻을 때까지 뽑기: p = 1/4, 최대 4번 / 90%를 넘기는 k / 제한 없이
E4 p를 바꾸면(1/10, 1/4, 1/2, 모두 가정)
E5 조각 영상: 두 조각을 한꺼번에 다시 뽑을 때와 조각마다 따로 다시 뽑을 때
E6 대조: 닫힌 식 / 하나씩 더하기 / 몬테카를로(시드 7, 20만 번)
E7 이미지: 보고서 표지 그림 1장(1024x1024 안팎), p = 1/4이면
E8 잡음 제거 장난감: 손계산, 시드 고정 재현, 단계 수, 마음에 드는 쪽으로 떨어지는 비율과 경계

표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import math
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch07_cost as C  # noqa: E402
import tools_ch07_diffusion as D  # noqa: E402

OUT = HERE / "results" / "ch07.json"
TARGET = 15
P_DEFAULT = F(1, 4)
K_DEFAULT = 4
MC_SEED = 7
MC_TRIALS = 200_000


def f(x, nd=6):
    return round(float(x), nd)


def e0_hand():
    c, p = 2000, P_DEFAULT
    rows = []
    for i in range(1, K_DEFAULT + 1):
        reach = (1 - p) ** (i - 1)            # i번째 뽑기를 하게 될 확률
        rows.append({"draw": i, "reach_prob": str(reach), "reach_prob_f": f(reach, 4),
                     "success_within": str(C.success_within(p, i)), "success_within_f": f(C.success_within(p, i), 4)})
    ed = C.expected_draws(p, K_DEFAULT)
    return {"cost_per_draw_krw": c, "p": str(p), "k": K_DEFAULT, "rows": rows,
            "expected_draws": str(ed), "expected_draws_f": f(ed, 4),
            "expected_cost_krw": f(ed * c, 1), "batch_cost_krw": c * K_DEFAULT,
            "unlimited_expected_draws": 4, "unlimited_expected_cost_krw": f(C.expected_cost_unlimited(c, p), 1),
            "k90": C.k_for_success(p, F(9, 10)),
            "success_k8": f(C.success_within(p, 8), 4), "success_k9": f(C.success_within(p, 9), 4)}


def e1_units(prices):
    omni = C.video_model("gemini-omni-1.1-flash", prices)
    imgs = []
    for img in prices["image"]:
        cost = C.image_cost(img)
        imgs.append({"label": img["label"], "size": img["size"],
                     "unit": "usd" if "usd_per_image" in img else ("tokens" if "tokens_per_image" in img else "credits"),
                     "amount": img.get("tokens_per_image", img.get("credits_per_image", img.get("usd_per_image"))),
                     "usd_per_image": f(cost, 6), "doc": img["usd_per_image_doc"]})
    return {"omni_tokens_per_second": omni["token_billing"]["tokens_per_second_720p"],
            "omni_usd_per_second": f(C.omni_usd_per_second(omni), 6),
            "omni_usd_15s": f(C.omni_usd_per_second(omni) * 15, 4),
            "images": imgs}


def e2_plans(prices):
    rows = []
    for m in prices["video"]:
        if not C.calculable(m):
            rows.append({"model": m["model"], "label": m["label"], "calculable": False,
                         "why": "API 종료" if m["api"] == "shut_down" else "한 번에 만드는 길이 선택지를 문서에서 못 찾음"})
            continue
        for res in ("720p", "1080p"):
            if res not in m["durations_by_res"]:
                continue
            plan = C.plan_clip(m, TARGET, res)
            c = C.cost_per_draw(m, plan, res)
            rows.append({"model": m["model"], "label": m["label"], "calculable": True, "res": res,
                         "how": plan["how"], "pieces": plan["pieces"], "billed_seconds": plan["billed_seconds"],
                         "usd_per_second": m["usd_per_second"][res], "usd_per_draw": f(c, 4)})
    return rows


def e3_draws(prices, plans):
    rows = []
    for r in plans:
        if not r.get("calculable") or r["res"] != "720p":
            continue
        c = F(str(r["usd_per_draw"]))
        k90 = C.k_for_success(P_DEFAULT, F(9, 10))
        rows.append({"label": r["label"], "usd_per_draw": f(c, 4),
                     "k4_expected": f(C.expected_cost_sequential(c, P_DEFAULT, K_DEFAULT), 4),
                     "k4_success": f(C.success_within(P_DEFAULT, K_DEFAULT), 4),
                     "k4_batch": f(C.cost_batch(c, K_DEFAULT), 4),
                     "k90": k90, "k90_worst": f(c * k90, 4),
                     "k90_expected": f(C.expected_cost_sequential(c, P_DEFAULT, k90), 4),
                     "unlimited": f(C.expected_cost_unlimited(c, P_DEFAULT), 4)})
    return rows


def e4_sensitivity(plans):
    rows = []
    for r in plans:
        if not r.get("calculable") or r["res"] != "720p":
            continue
        c = F(str(r["usd_per_draw"]))
        row = {"label": r["label"]}
        for p in (F(1, 10), F(1, 4), F(1, 2)):
            row[f"p={p}"] = f(C.expected_cost_unlimited(c, p), 4)
            row[f"k90@p={p}"] = C.k_for_success(p, F(9, 10))
        rows.append(row)
    return rows


def e5_pieces(prices):
    rows = []
    for model, res in (("veo-3.1-generate-preview", "720p"), ("veo-3.1-lite-generate-preview", "720p"),
                       ("gen4.5", "720p"), ("veo-3.1-generate-preview", "1080p")):
        m = C.video_model(model, prices)
        plan = C.plan_clip(m, TARGET, res)
        costs = C.piece_costs(m, plan, res)
        rows.append({"label": m["label"], "res": res, "how": plan["how"], "pieces": plan["pieces"],
                     "piece_costs": [f(x, 4) for x in costs],
                     "together": f(C.expected_cost_pieces_together(costs, P_DEFAULT), 4),
                     "together_success_per_draw": str(P_DEFAULT ** len(costs)),
                     "separately": f(C.expected_cost_pieces_separately(costs, P_DEFAULT), 4),
                     "separately_k4_both_success": f(C.success_within(P_DEFAULT, K_DEFAULT) ** len(costs), 4),
                     "separately_k90_per_piece": next(k for k in range(1, 100)
                                                       if C.success_within(P_DEFAULT, k) ** len(costs) >= F(9, 10))})
    return rows


def e6_crosscheck():
    bad = 0
    cases = 0
    for p in (F(1, 10), F(1, 4), F(1, 3), F(1, 2), F(9, 10)):
        for k in range(1, 21):
            cases += 1
            if C.expected_draws(p, k) != C.expected_draws_closed(p, k):
                bad += 1
    c = F("1.89")
    mc_cost, mc_ok = C.simulate_sequential(c, P_DEFAULT, K_DEFAULT, MC_TRIALS, MC_SEED)
    return {"exact_cases": cases, "exact_mismatches": bad,
            "mc_seed": MC_SEED, "mc_trials": MC_TRIALS, "mc_usd_per_draw": f(c, 4),
            "mc_expected_cost": f(mc_cost, 4), "closed_expected_cost": f(C.expected_cost_sequential(c, P_DEFAULT, K_DEFAULT), 4),
            "mc_success": f(mc_ok, 4), "closed_success": f(C.success_within(P_DEFAULT, K_DEFAULT), 4)}


def e7_images(prices):
    rows = []
    for img in prices["image"]:
        c = C.image_cost(img)
        rows.append({"label": img["label"], "size": img["size"], "usd_per_image": f(c, 5),
                     "unlimited_p_quarter": f(C.expected_cost_unlimited(c, P_DEFAULT), 5),
                     "k4_expected": f(C.expected_cost_sequential(c, P_DEFAULT, K_DEFAULT), 5)})
    return rows


def e8_diffusion():
    # 손계산: 개념 카드와 같은 0.9, 0.8
    keep, noise = D.two_step_noise_share(0.9, 0.8)
    x0, eps, abar = 1.0, -1.0, 0.72
    xt = D.forward(x0, abar, eps)
    back = (xt - math.sqrt(1 - abar) * eps) / math.sqrt(abar)
    hand = {"keep": round(keep, 4), "noise": round(noise, 4), "x0": x0, "eps": eps,
            "sqrt_abar": round(math.sqrt(abar), 4), "sqrt_1m_abar": round(math.sqrt(1 - abar), 4),
            "xt": round(xt, 4), "back": round(back, 4)}
    # 시드 고정 재현
    runs = []
    for s in range(8):
        x, path = D.generate(s, 50, trace=True)
        runs.append({"seed": s, "start": round(path[0], 4), "end": round(x, 4),
                     "mid": [round(path[i], 4) for i in (10, 25, 40)]})
    same = D.generate(3, 50) == D.generate(3, 50)
    x5, path5 = D.generate(1, 5, trace=True)
    # 한 번에 걷어내면
    one_step = [round(D.generate(s, 1), 4) for s in range(4)]
    # 단계 수
    seeds = list(range(2000))
    steps = [{"T": T, "settled": D.settled_fraction(seeds, T), "liked": D.liked_fraction(seeds, T)}
             for T in (1, 2, 3, 5, 10, 20, 50)]
    b = D.boundary(50)
    upper = 0.5 * (1 - math.erf(b / math.sqrt(2)))
    return {"hand": hand, "runs_T50": runs, "same_seed_same_result": same,
            "seed1_T5_path": [round(v, 4) for v in path5],
            "one_step_results": one_step, "prior_mean": sum(m * w for m, w in zip(D.MODES, D.WEIGHTS)),
            "steps": steps, "boundary_T50": round(b, 4), "liked_prob_from_boundary": round(upper, 4),
            "normal_q75": round(D.normal_upper_quantile(0.25), 4),
            "boundaries": {T: round(D.boundary(T), 4) for T in (5, 10, 50, 200)}}


def main():
    prices = C.load_prices()
    plans = e2_plans(prices)
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "mc_seed": MC_SEED, "price_file": C.PRICE_FILE.name, "checked": prices["checked"],
                 "target_seconds": TARGET, "p_default": str(P_DEFAULT), "k_default": K_DEFAULT},
        "e0_hand": e0_hand(),
        "e1_units": e1_units(prices),
        "e2_plans": plans,
        "e3_draws": e3_draws(prices, plans),
        "e4_sensitivity": e4_sensitivity(plans),
        "e5_pieces": e5_pieces(prices),
        "e6_crosscheck": e6_crosscheck(),
        "e7_images": e7_images(prices),
        "e8_diffusion": e8_diffusion(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
