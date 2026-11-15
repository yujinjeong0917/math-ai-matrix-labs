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


# 가격 값은 JSON에 문자열("0.126")로 들어 있다. float을 거치지 않고 Fraction(str)로 읽어야 0.1 + 0.2 같은 이진 부동소수점 오차가 생기지 않는다.
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

# 동적계획법(DP, 작은 합의 답으로 큰 합의 답을 만드는 방법). best[t] = 합이 정확히 t인 조합 중 조각 수가 가장 적은 것.
# 불변식: total을 처리할 때 total보다 작은 모든 합의 best가 이미 채워져 있다. 시간복잡도 O((target + max) * |choices|) (리스트 복사 비용 제외).
# 합이 정확히 target이 안 되는 길이 구성도 있으므로 target 이상의 가장 작은 합을 고른다(초과분도 과금된다는 가정).
def _min_sum_at_least(choices, target):
    """choices에서 (중복 허용) 골라 합이 target 이상이 되는 가장 작은 합과, 그때 조각 수가 가장 적은 조합."""
    best = {0: []}
    for total in range(1, target + max(choices) + 1):
        cands = [best[total - d] + [d] for d in choices if total - d in best]
        if cands:
            # 동률이면 조각 수가 적은 쪽, 그다음 긴 조각이 앞서는 쪽을 고른다(모형의 선택).
            best[total] = min(cands, key=lambda c: (len(c), [-x for x in sorted(c, reverse=True)]))
    # target 이상인 합 중 가장 작은 값. 상한을 target + max(choices)로 둔 것은, 그보다 큰 합은 더 작은 합의 조각 하나를 빼도 target 이상이라 최소가 될 수 없기 때문.
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
    # 확장 방식: 기본 길이로 만든 뒤 확장 길이를 target 이상이 될 때까지 덧붙인다. 확장 조각은 새로 만든 초만큼만 과금된다는 가정(문서 해석, 확인 필요).
    ext = m.get("extension")
    if ext and res in ext.get("resolutions", []) and ext.get("base_seconds") in durs:
        pieces = [ext["base_seconds"]]
        while sum(pieces) < target:
            pieces.append(ext["seconds"])
        plans.append({"how": "extend", "pieces": pieces})
    # 한 번에 target초가 되는 선택지가 없을 때만 이어 붙이기를 후보에 넣는다. 한 번에 되는 길이가 있으면 이어 붙이기는 비교 대상에서 뺀다(모형의 단순화).
    if not longer:
        plans.append({"how": "stitch", "pieces": _min_sum_at_least(durs, target)})
    for p in plans:
        p["billed_seconds"] = sum(p["pieces"])
    # 과금 초가 적은 방법 우선, 같으면 조각 수가 적은 방법.
    return min(plans, key=lambda p: (p["billed_seconds"], len(p["pieces"])))


# 한 번 뽑기 값 = 초당 달러 x 과금 초. 초당 값이 문자열이라 frac으로 읽는다.
def cost_per_draw(m, plan, res="720p"):
    """한 번 뽑기(조각 모두 새로)의 값. 달러, Fraction."""
    return frac(m["usd_per_second"][res]) * plan["billed_seconds"]


def piece_costs(m, plan, res="720p"):
    rate = frac(m["usd_per_second"][res])
    return [rate * s for s in plan["pieces"]]


# ---------- 여러 번 뽑기 ----------

# 식: P(적어도 한 번 성공) = 1 - P(k번 모두 실패) = 1 - (1-p)^k. 뽑기끼리 독립이고 p가 매번 같다는 가정(실제로는 프롬프트를 고치면 p가 바뀐다).
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
    # i번째 뽑기를 하게 될 확률은 앞 i-1번이 모두 실패할 확률. 이 등비급수의 합이 아래 닫힌 식과 같다.
    return sum((1 - p) ** (i - 1) for i in range(1, k + 1))


def expected_draws_closed(p, k):
    p = F(p)
    # 등비급수 합: sum_{j=0}^{k-1} (1-p)^j = (1 - (1-p)^k) / p. k -> 무한대에서 1/p.
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
    # 불변식: 루프에 들어올 때 success_within(p, k) < target (k-1까지는 모자람). 정확한 분수 비교라 경계값(예: 0.9를 딱 맞출 때)에서도 틀리지 않는다. p = 0이면 끝나지 않는다.
    while success_within(p, k) < target:
        k += 1
    return k


def expected_cost_pieces_together(costs, p):
    """조각을 한꺼번에 다시 뽑고, 모든 조각이 동시에 마음에 들어야 성공. 성공 확률 p^조각수."""
    # 조각을 한꺼번에 다시 뽑으면 한 번의 성공 확률이 p^조각수. 성공까지의 기대 뽑기 수 = 1/p^조각수이므로 비용 = (한 번 뽑기 값 합) / p^조각수.
    c = sum(costs, F(0))
    return c / (F(p) ** len(costs))


def expected_cost_pieces_separately(costs, p):
    """조각마다 마음에 들 때까지 따로 다시 뽑음. 조각마다 c_i / p."""
    # 조각마다 따로 다시 뽑으면 조각 i의 기대 비용 c_i / p를 더한다. 같은 p라도 조각 수가 늘수록 together보다 훨씬 싸다.
    return sum((F(ci) / F(p) for ci in costs), F(0))


def simulate_sequential(c, p, k, trials, seed):
    """몬테카를로: 차례로 뽑다 멈추기를 trials번 흉내. (평균 비용, 성공 비율)."""
    # 시드 고정: 몬테카를로 결과가 매번 같아야 테스트의 허용 오차 검사가 안정적이다.
    rng = random.Random(seed)
    total = 0.0
    ok = 0
    for _ in range(trials):
        for i in range(1, k + 1):
            # 뽑을 때마다 비용을 낸다(성공해도 그 뽑기 값은 낸다). 성공하면 멈춘다.
            total += float(c)
            if rng.random() < float(p):
                ok += 1
                break
    return total / trials, ok / trials


# ---------- 이미지 ----------

# 이미지 값은 세 가지 단위(장당 달러 / 토큰 x 100만 토큰당 달러 / 크레딧 x 크레딧당 달러)를 모두 달러로 환산한다.
def image_cost(img):
    """장당 값. 문서가 장당 값만 적으면 그 값, 토큰 과금이면 토큰 수 x 100만 토큰당 값, 크레딧이면 크레딧 x 크레딧 값."""
    if "usd_per_image" in img:
        return frac(img["usd_per_image"])
    if "tokens_per_image" in img:
        return F(img["tokens_per_image"]) * frac(img["usd_per_mtok"]) / 1_000_000
    return F(img["credits_per_image"]) * frac(img["usd_per_credit"])


def omni_usd_per_second(m):
    tb = m["token_billing"]
    # 초당 토큰 수 x 100만 토큰당 달러 / 1,000,000. 토큰 과금도 결국 초당 달러로 바뀐다.
    return F(tb["tokens_per_second_720p"]) * frac(tb["usd_per_mtok_video_output"]) / 1_000_000
