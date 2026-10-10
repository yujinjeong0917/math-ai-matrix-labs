"""AI 도구 실전 7장 검증. `uv run pytest -q tools/ch07_image_video` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import json
import math
from fractions import Fraction as F
from pathlib import Path

import pytest

import tools_ch07_cost as C
import tools_ch07_diffusion as D

HERE = Path(__file__).parent


# ---------- 가격표 ----------

def test_price_file_has_source_and_date_for_every_entry():
    prices = C.load_prices()
    assert prices["checked"] == "2026-10-10"
    assert C.PRICE_FILE.name == "prices_2026-10-10.json"
    for group in ("video", "image"):
        for m in prices[group]:
            assert m["url"].startswith("https://") and m["url_spec"].startswith("https://")
            assert m["checked"] == "2026-10-10"
    for t in prices["terms"]:
        assert t["checked"] == "2026-10-10"


def test_doc_numbers_copied_correctly():
    v = lambda name: C.video_model(name)  # noqa: E731
    assert v("veo-3.1-generate-preview")["usd_per_second"] == {"720p": "0.40", "1080p": "0.40", "4k": "0.60"}
    assert v("veo-3.1-fast-generate-preview")["usd_per_second"]["720p"] == "0.10"
    assert v("veo-3.1-lite-generate-preview")["usd_per_second"]["720p"] == "0.05"
    assert v("kling-3.0")["usd_per_second"]["720p"] == "0.126"
    assert v("kling-3.0-silent")["usd_per_second"]["720p"] == "0.084"
    assert v("gen4.5")["usd_per_second"]["720p"] == "0.12"
    assert v("sora-2")["api"] == "shut_down" and v("sora-2")["shutdown_date"] == "2026-09-24"
    # Kling: 1 Unit = 0.14달러, 소리 켬 720p 0.9 Unit
    assert F("0.9") * F("0.14") == F("0.126")
    # Runway: 12 크레딧 x 0.01달러
    assert 12 * F("0.01") == F("0.12")


def test_token_billing_converts_to_seconds_and_images():
    omni = C.video_model("gemini-omni-1.1-flash")
    assert C.omni_usd_per_second(omni) == F("0.10136")          # 문서: 약 0.10달러/초
    prices = C.load_prices()
    for img in prices["image"]:
        # 문서가 적은 장당 값과 3자리 안에서 같다
        assert abs(float(C.image_cost(img)) - float(img["usd_per_image_doc"])) < 0.0006


# ---------- 15초 채우기 ----------

@pytest.mark.parametrize("model,res,how,pieces", [
    ("veo-3.1-generate-preview", "720p", "extend", [8, 7]),
    ("veo-3.1-generate-preview", "1080p", "stitch", [8, 8]),
    ("veo-3.1-fast-generate-preview", "720p", "extend", [8, 7]),
    ("veo-3.1-lite-generate-preview", "720p", "stitch", [8, 8]),
    ("kling-3.0", "720p", "single", [15]),
    ("gen4.5", "720p", "stitch", [10, 5]),
])
def test_plan_for_15_seconds(model, res, how, pieces):
    plan = C.plan_clip(C.video_model(model), 15, res)
    assert plan["how"] == how and plan["pieces"] == pieces and plan["billed_seconds"] == sum(pieces)


def test_plan_never_bills_less_than_target_and_is_minimal():
    prices = C.load_prices()
    for m in prices["video"]:
        if not C.calculable(m):
            continue
        for res, durs in m["durations_by_res"].items():
            for target in range(1, 31):
                plan = C.plan_clip(m, target, res)
                assert plan["billed_seconds"] >= target
                # 모든 조각이 허용된 길이(확장 조각은 확장 길이)
                ext = m.get("extension") or {}
                for i, s in enumerate(plan["pieces"]):
                    assert s in durs or (plan["how"] == "extend" and i > 0 and s == ext["seconds"])


def test_uncalculable_models_are_excluded():
    assert not C.calculable(C.video_model("sora-2"))
    assert not C.calculable(C.video_model("gemini-omni-1.1-flash"))


# ---------- 여러 번 뽑기 ----------

def test_first_screen_hand_numbers():
    p = F(1, 4)
    assert [C.success_within(p, k) for k in (1, 2, 3, 4)] == [F(1, 4), F(7, 16), F(37, 64), F(175, 256)]
    assert C.expected_draws(p, 4) == F(175, 64) == 1 + F(3, 4) + F(9, 16) + F(27, 64)
    assert C.expected_cost_sequential(2000, p, 4) == F(2000 * 175, 64)        # 5,468.75원
    assert C.cost_batch(2000, 4) == 8000
    assert C.expected_cost_unlimited(2000, p) == 8000


def test_k_for_90_percent_is_exact():
    p = F(1, 4)
    assert C.success_within(p, 8) < F(9, 10) < C.success_within(p, 9)
    assert C.k_for_success(p, F(9, 10)) == 9
    assert C.k_for_success(F(1, 10), F(9, 10)) == 22
    assert C.k_for_success(F(1, 2), F(9, 10)) == 4


@pytest.mark.parametrize("p", [F(1, 10), F(1, 4), F(1, 3), F(1, 2), F(9, 10)])
def test_closed_form_matches_sum(p):
    for k in range(1, 25):
        assert C.expected_draws(p, k) == C.expected_draws_closed(p, k)
        assert C.expected_draws(p, k) <= k
    # k가 커지면 1/p로 다가간다
    assert abs(float(C.expected_draws(p, 400)) - float(1 / p)) < 1e-9


def test_monte_carlo_agrees():
    c, p, k = F("1.89"), F(1, 4), 4
    mc_cost, mc_ok = C.simulate_sequential(c, p, k, 200_000, 7)
    assert abs(mc_cost - float(C.expected_cost_sequential(c, p, k))) < 0.02
    assert abs(mc_ok - float(C.success_within(p, k))) < 0.005


def test_pieces_together_vs_separately():
    m = C.video_model("gen4.5")
    plan = C.plan_clip(m, 15, "720p")
    costs = C.piece_costs(m, plan, "720p")
    p = F(1, 4)
    assert C.expected_cost_pieces_separately(costs, p) == F("7.2")
    assert C.expected_cost_pieces_together(costs, p) == F("28.8")      # 1/16 확률이라 4배
    assert C.expected_cost_pieces_together(costs, p) == 4 * C.expected_cost_pieces_separately(costs, p)


def test_results_file_matches_modules():
    res = json.loads((HERE / "results" / "ch07.json").read_text(encoding="utf-8"))
    e0 = res["e0_hand"]
    assert e0["expected_draws"] == "175/64" and e0["expected_cost_krw"] == 5468.8 and e0["k90"] == 9
    draws = {r["label"]: r for r in res["e3_draws"]}
    assert draws["Veo 3.1"]["usd_per_draw"] == 6.0 and draws["Veo 3.1"]["unlimited"] == 24.0
    assert draws["Kling 3.0 (소리 켬)"]["k4_expected"] == round(float(F("1.89") * F(175, 64)), 4)
    assert draws["Runway Gen-4.5"]["usd_per_draw"] == 1.8
    assert res["e6_crosscheck"]["exact_mismatches"] == 0


# ---------- 잡음 제거 장난감 ----------

def test_hand_forward_and_back():
    keep, noise = D.two_step_noise_share(0.9, 0.8)
    assert round(keep, 4) == 0.72 and round(noise, 4) == 0.28
    xt = D.forward(1.0, 0.72, -1.0)
    assert round(xt, 4) == 0.3194
    assert math.isclose((xt + math.sqrt(0.28)) / math.sqrt(0.72), 1.0)


def test_same_seed_same_result_different_seed_differs():
    assert D.generate(3, 50) == D.generate(3, 50)
    ends = {round(D.generate(s, 50), 3) for s in range(8)}
    assert ends == {-1.0, 1.0}


def test_one_step_gives_blurry_average():
    for s in range(20):
        assert abs(D.generate(s, 1) - (-0.5)) < 0.02          # 두 그림의 평균 근처, 어느 쪽도 아님


def test_more_steps_settle_and_liked_share_near_quarter():
    seeds = range(2000)
    assert D.settled_fraction(seeds, 1) == 0.0
    assert D.settled_fraction(seeds, 50) == 1.0
    assert abs(D.liked_fraction(seeds, 50) - 0.25) < 0.03
    b = D.boundary(50)
    assert abs(b - D.normal_upper_quantile(0.25)) < 0.01
    assert abs(D.boundary(200) - D.normal_upper_quantile(0.25)) < abs(D.boundary(5) - D.normal_upper_quantile(0.25))
