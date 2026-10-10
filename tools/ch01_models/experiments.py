"""AI 도구 실전 1장 실험. `uv run python tools/ch01_models/experiments.py` 로 실행한다.

E0 첫 화면 손계산: Claude Sonnet 5.5로 한 번 호출, 한 주(12번), 한 달(52/12주)
E1 비교표: 가격표의 모델마다 문맥 창, 최대 출력, 단가, 가중치, 데이터 정책, 학습 기준일
E2 한 달 비용(캐싱 없음, 생각 토큰 없음): 모델별, 2장 표의 "출력 포함 캐싱 없음" 값과 대조
E3 숨은 출력: 호출마다 생각 토큰 2,000개(가정)가 더 붙을 때
E4 긴 프롬프트 구간: 분기 보고서(문의 13주치 312,000토큰)를 12번 부를 때 걸리는 규칙
E5 시간대: DeepSeek 피크를 한국 시각으로 옮기기, 월요일 10시와 20시에 돌릴 때
E6 팀 조건으로 거르기
E7 답이 갈리는 이유 장난감: 온도를 넣은 소프트맥스, 1,000번 뽑기, 두 모델의 1등
E8 대조: 무작위 500경우에서 계산기와 따로 짠 계산이 같은지

표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch01_answers as A  # noqa: E402
import tools_ch01_cost as C  # noqa: E402

OUT = HERE / "results" / "ch01.json"
SEED = 1


def f(x, nd=6):
    return round(float(x), nd)


def e0_hand(w):
    m = C.price_of("claude-sonnet-5-5")
    it = C.input_tokens_per_call(w)
    ot = w["output_tokens_per_call"]
    call_in = F(it) * C.frac(m["input"]) / C.MTOK
    call_out = F(ot) * C.frac(m["output"]) / C.MTOK
    call = call_in + call_out
    week = call * w["calls_per_week"]
    month = week * C.weeks_per_month(w)
    return {"input_tokens": it, "output_tokens": ot, "call_input": f(call_in), "call_output": f(call_out),
            "call": f(call), "calls_per_week": w["calls_per_week"], "week": f(week),
            "weeks_per_month": f(C.weeks_per_month(w), 4), "month": f(month, 4), "year": f(week * 52, 4)}


def e1_table(prices):
    rows = []
    for m in C.sorted_models(prices):
        rows.append({"provider": m["provider"], "model": m["model"], "label": m["label"],
                     "context": m["context"], "max_output": m["max_output"],
                     "input": m["input"], "output": m["output"],
                     "offpeak": None if not m["peak"] else [m["peak"]["offpeak_input"], m["peak"]["offpeak_output"]],
                     "long_rule": None if not m["long"] else m["long"]["rule"],
                     "weights_public": m["weights"]["public"], "license": m["weights"]["license"],
                     "trains_on_api": m["data"]["trains_on_api"], "knowledge_cutoff": m["knowledge_cutoff"]})
    return rows


def e2_month(prices, w):
    rows = []
    for m in C.sorted_models(prices):
        row = {"model": m["model"], "week": f(C.week_cost(m, w)), "month": f(C.month_cost(m, w), 4),
               "year": f(C.week_cost(m, w) * 52, 4), "rule": C.unit_prices(m, C.input_tokens_per_call(w))[2]}
        if m["peak"]:
            row["week_offpeak"] = f(C.week_cost(m, w, peak=False))
            row["month_offpeak"] = f(C.month_cost(m, w, peak=False), 4)
        rows.append(row)
    return rows


def e3_thinking(prices, w):
    t = w["thinking_tokens_assumed"]
    rows = []
    for m in C.sorted_models(prices):
        a, b = C.month_cost(m, w), C.month_cost(m, w, thinking=t)
        rows.append({"model": m["model"], "month": f(a, 4), "month_thinking": f(b, 4), "ratio": f(b / a, 4),
                     "documented": m["thinking_billed_as_output"]})
    return {"thinking_tokens": t, "rows": rows}


def e4_quarter(prices, w):
    q = w["quarter_weeks"]
    it = C.input_tokens_per_call(w, q)
    ot = w["output_tokens_per_call"]
    n = w["calls_per_week"]
    rows = []
    for m in C.sorted_models(prices):
        i, o, rule = C.unit_prices(m, it)
        run = n * C.call_cost(m, it, ot)
        base = n * (F(it) * C.frac(m["input"]) + F(ot) * C.frac(m["output"])) / C.MTOK
        rows.append({"model": m["model"], "rule": rule, "unit_in": f(i, 4), "unit_out": f(o, 4),
                     "run": f(run, 4), "if_base_rate": f(base, 4), "ratio": f(run / base, 4),
                     "fits": C.fits(m, it, ot)})
    return {"input_tokens": it, "output_tokens": ot, "calls": n, "rows": rows}


def e5_peak(prices, w):
    days = ["월", "화", "수", "목", "금", "토", "일"]
    peak_hours = {d: [h for h in range(24) if C.is_peak_kst(i, h)] for i, d in enumerate(days)}
    out = {"peak_hours_kst": peak_hours}
    for name in ("deepseek-v4-pro", "deepseek-flash"):
        m = C.price_of(name, prices)
        out[name] = {"mon_10": f(C.month_cost(m, w, peak=C.is_peak_kst(0, 10)), 4),
                     "mon_20": f(C.month_cost(m, w, peak=C.is_peak_kst(0, 20)), 4)}
    return out


def e6_choose(prices, w):
    rows = []
    for m in C.sorted_models(prices):
        c = C.check_conditions(m, w)
        status = "통과" if all(v is True for v in c.values()) else ("보류" if None in c.values() and False not in c.values() else "탈락")
        rows.append({"model": m["model"], **c, "status": status})
    return rows


def e7_answers():
    out = {"question": A.QUESTION, "model_a": A.MODEL_A, "model_b": A.MODEL_B}
    for T in (1.0, 0.5):
        p = A.softmax(A.MODEL_A, T)
        counts, _ = A.sample(A.MODEL_A, T, 1000, SEED)
        hr = A.hand_ratios(A.MODEL_A, T)
        out[f"T{T}"] = {"p": {k: round(v, 3) for k, v in p.items()}, "counts": counts,
                        "hand": {k: round(v, 3) for k, v in hr.items()}, "hand_sum": round(sum(hr.values()), 3)}
    _, seq = A.sample(A.MODEL_A, 1.0, 5, SEED)
    out["five_runs_T1"] = seq
    out["greedy_a"] = A.greedy(A.MODEL_A)
    out["greedy_b"] = A.greedy(A.MODEL_B)
    out["b_T1"] = {k: round(v, 3) for k, v in A.softmax(A.MODEL_B, 1.0).items()}
    return out


def independent_cost(m, it, ot, peak):
    """계산기와 따로 짠 계산: 회사별 규칙을 if 문으로 직접 쓴다."""
    p = m["provider"]
    i, o = F(str(m["input"])), F(str(m["output"]))
    if p == "DeepSeek" and not peak:
        i, o = i / 2, o / 2                          # 문서: 비피크는 피크의 절반
    if m["model"] == "claude-haiku-5-5" and it > 100_000:
        i, o = F(1, 2), F(5, 2)
    if p == "OpenAI" and it > 272_000:
        i, o = i * 2, o * F(3, 2)
    if p == "xAI" and it >= 200_000:
        i, o = F(4), F(12)
    total = F(0)
    total += it * i
    total += ot * o
    return total / 1_000_000


def e8_crosscheck(prices):
    rng = random.Random(SEED)
    bad = 0
    for _ in range(500):
        m = rng.choice(prices["models"])
        it = rng.randint(1_000, 480_000)
        ot = rng.randint(1, 20_000)
        peak = rng.random() < 0.5
        if C.call_cost(m, it, ot, peak) != independent_cost(m, it, ot, peak):
            bad += 1
    return {"cases": 500, "mismatches": bad, "seed": SEED}


def main():
    prices, w = C.load_prices(), C.load_workload()
    res = {"meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    "seed": SEED, "price_file": C.PRICE_FILE.name, "checked": prices["checked"]},
           "e0_hand": e0_hand(w), "e1_table": e1_table(prices), "e2_month": e2_month(prices, w),
           "e3_thinking": e3_thinking(prices, w), "e4_quarter": e4_quarter(prices, w),
           "e5_peak": e5_peak(prices, w), "e6_choose": e6_choose(prices, w), "e7_answers": e7_answers(),
           "e8_crosscheck": e8_crosscheck(prices)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("e0_hand", "e2_month", "e6_choose", "e8_crosscheck")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
