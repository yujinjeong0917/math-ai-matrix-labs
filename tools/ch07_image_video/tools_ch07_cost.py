"""AI 도구 실전 7장: 영상·이미지 생성 비용 계산.

- 가격표 읽기(data/prices_*.json)
- 15초짜리 영상을 모델이 허락하는 길이로 어떻게 채우나(한 번에, 확장, 여러 조각 이어 붙이기)와 과금 초
- 마음에 드는 것 하나를 얻을 때까지 뽑기: 성공 확률 p(가정값), 최대 k번
  * 차례로 뽑다가 마음에 들면 멈춤: 기대 비용 = c * (1 - (1-p)^k) / p, 성공 확률 = 1 - (1-p)^k
  * 한꺼번에 k개 뽑기: 비용 = k * c, 성공 확률은 같다
  * 제한 없이 될 때까지: 기대 비용 = c / p

표준 라이브러리만 쓴다. 금액과 확률은 Fraction으로 계산해 반올림 오차가 없다.
"""

import json
import random
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
PRICE_FILE = HERE / "data" / "prices_2026-10-10.json"


def frac(x):
    return None if x is None else F(str(x))


def load_prices(path=PRICE_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def video_model(model, prices=None):
    prices = prices or load_prices()
    for m in prices["video"]:
        if m["model"] == model:
            return m
    raise KeyError(model)


def calculable(m):
    """한 번에 만드는 길이의 선택지와 초당 값이 문서에 있어 계산기에 넣을 수 있는 모델인가."""
    return m["api"] == "available" and m["durations_by_res"] is not None and m["usd_per_second"] is not None


# ---------- 15초를 어떻게 채우나 ----------

def _min_sum_at_least(choices, target):
    """choices에서 (중복 허용) 골라 합이 target 이상이 되는 가장 작은 합과, 그때 조각 수가 가장 적은 조합."""
    best = {0: []}
    for total in range(1, target + max(choices) + 1):
        cands = [best[total - d] + [d] for d in choices if total - d in best]
        if cands:
            best[total] = min(cands, key=lambda c: (len(c), [-x for x in sorted(c, reverse=True)]))
    totals = sorted(t for t in best if t >= target)
    t = totals[0]
    return sorted(best[t], reverse=True)


def plan_clip(m, target=15, res="720p"):
    """target초 영상을 만드는 방법. 과금 초가 가장 적은 방법, 같으면 조각 수가 적은 방법.

    반환: {"how": "single" | "extend" | "stitch", "pieces": [초, ...], "billed_seconds": 합}
    - single: 한 번 생성으로 target초 이상(가장 짧은 것)
    - extend: 기본 길이로 만든 뒤 확장을 n번(확장은 새로 만든 초만큼 낸다고 가정)
    - stitch: 따로 만든 조각을 이어 붙임
    """
    durs = m["durations_by_res"].get(res)
    if durs is None:
        raise ValueError(f"{m['model']}는 {res}를 지원하지 않음")
    plans = []
    longer = [d for d in durs if d >= target]
    if longer:
        plans.append({"how": "single", "pieces": [min(longer)]})
    ext = m.get("extension")
    if ext and res in ext.get("resolutions", []) and ext.get("base_seconds") in durs:
        pieces = [ext["base_seconds"]]
        while sum(pieces) < target:
            pieces.append(ext["seconds"])
        plans.append({"how": "extend", "pieces": pieces})
    if not longer:
        plans.append({"how": "stitch", "pieces": _min_sum_at_least(durs, target)})
    for p in plans:
        p["billed_seconds"] = sum(p["pieces"])
    return min(plans, key=lambda p: (p["billed_seconds"], len(p["pieces"])))


def cost_per_draw(m, plan, res="720p"):
    """한 번 뽑기(조각 모두 새로)의 값. 달러, Fraction."""
    return frac(m["usd_per_second"][res]) * plan["billed_seconds"]


def piece_costs(m, plan, res="720p"):
    rate = frac(m["usd_per_second"][res])
    return [rate * s for s in plan["pieces"]]


# ---------- 여러 번 뽑기 ----------

def success_within(p, k):
    """최대 k번 뽑아 한 번이라도 마음에 들 확률 1 - (1-p)^k."""
    p = F(p)
    return 1 - (1 - p) ** k


def expected_draws(p, k):
    """마음에 들면 멈추고 최대 k번까지 뽑을 때 실제로 뽑는 횟수의 기댓값.

    i번째 뽑기를 하게 될 확률 = 앞의 i-1번이 모두 마음에 안 들 확률 (1-p)^(i-1).
    더하면 (1 - (1-p)^k) / p.
    """
    p = F(p)
    return sum((1 - p) ** (i - 1) for i in range(1, k + 1))


def expected_draws_closed(p, k):
    p = F(p)
    return (1 - (1 - p) ** k) / p


def expected_cost_sequential(c, p, k):
    return F(c) * expected_draws_closed(p, k)


def cost_batch(c, k):
    return F(c) * k


def expected_cost_unlimited(c, p):
    return F(c) / F(p)


def k_for_success(p, target):
    """1 - (1-p)^k >= target 이 되는 가장 작은 k (정확한 분수 비교)."""
    p, target = F(p), F(target)
    k = 1
    while success_within(p, k) < target:
        k += 1
    return k


def expected_cost_pieces_together(costs, p):
    """조각을 한꺼번에 다시 뽑고, 모든 조각이 동시에 마음에 들어야 성공. 성공 확률 p^조각수."""
    c = sum(costs, F(0))
    return c / (F(p) ** len(costs))


def expected_cost_pieces_separately(costs, p):
    """조각마다 마음에 들 때까지 따로 다시 뽑음. 조각마다 c_i / p."""
    return sum((F(ci) / F(p) for ci in costs), F(0))


def simulate_sequential(c, p, k, trials, seed):
    """몬테카를로: 차례로 뽑다 멈추기를 trials번 흉내. (평균 비용, 성공 비율)."""
    rng = random.Random(seed)
    total = 0.0
    ok = 0
    for _ in range(trials):
        for i in range(1, k + 1):
            total += float(c)
            if rng.random() < float(p):
                ok += 1
                break
    return total / trials, ok / trials


# ---------- 이미지 ----------

def image_cost(img):
    """장당 값. 문서가 장당 값만 적으면 그 값, 토큰 과금이면 토큰 수 x 100만 토큰당 값, 크레딧이면 크레딧 x 크레딧 값."""
    if "usd_per_image" in img:
        return frac(img["usd_per_image"])
    if "tokens_per_image" in img:
        return F(img["tokens_per_image"]) * frac(img["usd_per_mtok"]) / 1_000_000
    return F(img["credits_per_image"]) * frac(img["usd_per_credit"])


def omni_usd_per_second(m):
    tb = m["token_billing"]
    return F(tb["tokens_per_second_720p"]) * frac(tb["usd_per_mtok_video_output"]) / 1_000_000
