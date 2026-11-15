"""0장 실습: 예측과 개입, 반사실. 모든 숫자는 가상(예시용) 장난감이며 시드를 고정했다.

난수는 random.Random(seed).random() 만 쓴다(파이썬 버전이 달라도 같은 값).

무엇을 재는가
- 장면 A: 관찰 데이터의 "본 사람 vs 안 본 사람" 평균 격차와, 노출을 동전으로 정했을 때의 격차를 진짜 효과(0.3)와 비교한다.
- 장면 B: 새 기능 도입 전후의 단순 비교와, 작년 같은 주 창을 한 번 더 빼는 "계절 반사실" 어림값을 진짜 효과(+5)와 비교한다.
입력: scene_a(n, seed), scene_b(other_event, seed). 외부 파일 없이 난수로 데이터를 만든다.
출력: 격차·어림값을 담은 dict. 테스트가 소수 둘째 자리까지 고정해 둔 숫자이다(장 본문 숫자와의 대응은 README로 확인 필요)
(관찰 격차 2.29, 동전 배정 격차 0.27, 단순 전후 15.01, 계절 반사실 4.46, 광고 증액 시 16.46).
"""
import math
import random


def normal(rng, sd=1.0):
    """Box-Muller. rng.random() 두 번으로 표준정규 한 개를 만든다."""
    # random()은 [0, 1)이라 0이 나올 수 있고 log(0)은 정의되지 않는다. 1 - u로 (0, 1]로 옮긴다.
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def mean(xs):
    """산술평균. 제너레이터도 받도록 먼저 리스트로 만든다(빈 입력이면 ZeroDivisionError)."""
    xs = list(xs)
    return sum(xs) / len(xs)


# ---------- 장면 A: 위젯을 본 사람이 더 산다 vs 위젯을 보여 주면 더 산다 ----------
WIDGET_TRUE_EFFECT = 0.3  # 가상: 위젯을 보면 월 주문이 0.3건 는다


def widget_orders(h, seen, rng):
    """월 주문 건수(건). 열성도 h가 6.0건/단위로 크게 작용하고, 위젯 효과는 0.3건뿐이다."""
    return 2.0 + 6.0 * h + WIDGET_TRUE_EFFECT * seen + normal(rng, 1.0)


def scene_a(n=20000, seed=7):
    """열성도 h(0~1)가 높은 사람일수록 위젯 영역까지 스크롤한다(관찰 데이터).
    같은 장난감에서 위젯 노출을 동전으로 정하면(무작위 개입) 격차가 어떻게 달라지는지 본다."""
    rng = random.Random(seed)
    obs_seen, obs_not = [], []
    for _ in range(n):
        h = rng.random()
        # 관찰 데이터: 노출 확률이 h 자체다. h가 주문과 노출 둘 다에 영향을 주는 교란이 여기서 생긴다.
        seen = 1 if rng.random() < h else 0
        y = widget_orders(h, seen, rng)
        (obs_seen if seen else obs_not).append(y)
    rng = random.Random(seed + 1)  # 시드를 달리해 두 세계의 난수열이 겹치지 않게 한다
    ran_seen, ran_not = [], []
    for _ in range(n):
        h = rng.random()
        # h와 무관한 확률 0.5라서 h가 두 집단에 고르게 섞인다(무작위 개입)
        seen = 1 if rng.random() < 0.5 else 0  # 동전으로 노출
        y = widget_orders(h, seen, rng)
        (ran_seen if seen else ran_not).append(y)
    return {
        "observed_gap": mean(obs_seen) - mean(obs_not),
        "randomized_gap": mean(ran_seen) - mean(ran_not),
        "true_effect": WIDGET_TRUE_EFFECT,
        "n_seen_obs": len(obs_seen),
        "n_not_obs": len(obs_not),
    }


# ---------- 장면 B: 계절이 같이 움직일 때 전후 비교와 계절 반사실 ----------
FEATURE_WEEK = 40          # 새 기능 도입 주(가상)
FEATURE_TRUE_EFFECT = 5.0  # 가상: 기능의 진짜 효과 주당 +5 (주문지수)
POST_WEEKS = range(40, 48)  # 도입 뒤 관찰 8주
PRE_WEEKS = range(32, 40)   # 도입 전 8주


def season(w):
    """주(w)별 계절 성분. 진폭 15, 주기 52주, w=34에서 0으로 올라가기 시작하는 사인 곡선(모형의 선택)."""
    return 15.0 * math.sin(2.0 * math.pi * (w - 34) / 52.0)


def make_weeks(seed=11, other_event=0.0, noise=2.0):
    """1년차(기능 없음) 52주와 2년차 47주를 만든다. 2년차는 기본 수준이 +10 더 높다.
    other_event: 도입 주와 같은 주에 시작한 다른 변화(예: 광고 증액)의 주당 효과."""
    rng = random.Random(seed)
    # 작년은 기능이 없어 반사실의 대용 데이터가 된다. 가정: 계절 모양이 해마다 같다.
    y1 = {w: 100.0 + season(w) + normal(rng, noise) for w in range(1, 53)}
    y2 = {}
    for w in range(1, 48):
        y = 110.0 + season(w) + normal(rng, noise)
        if w >= FEATURE_WEEK:  # 도입 주부터 기능 효과와 동시 사건 효과가 함께 얹힌다
            y += FEATURE_TRUE_EFFECT + other_event
        y2[w] = y
    return y1, y2


def naive_before_after(y2):
    """단순 전후 비교: 도입 후 8주 평균 - 도입 전 8주 평균. 계절 상승분까지 효과로 읽는 함정이 있다."""
    return mean(y2[w] for w in POST_WEEKS) - mean(y2[w] for w in PRE_WEEKS)


def window_means(y1, y2):
    """올해 도입 전·후 8주 평균과, 작년 같은 주 창의 평균(작년에는 기능이 없었다)."""
    return {
        "this_pre": mean(y2[w] for w in PRE_WEEKS), "this_post": mean(y2[w] for w in POST_WEEKS),
        "last_pre": mean(y1[w] for w in PRE_WEEKS), "last_post": mean(y1[w] for w in POST_WEEKS),
    }


def seasonal_counterfactual_effect(y1, y2):
    """올해 전후 변화에서, 작년 같은 주 창의 전후 변화(계절만의 변화)를 한 번 더 뺀다."""
    m = window_means(y1, y2)
    # 이중 차분 꼴: (올해 후 - 올해 전) - (작년 후 - 작년 전). 올해에만 다른 사건이 있으면 이 뺄셈으로도 못 거른다.
    return (m["this_post"] - m["this_pre"]) - (m["last_post"] - m["last_pre"]), m


def scene_b(other_event=0.0, seed=11):
    """other_event=0이면 계절만 움직이고, 12이면 같은 주에 광고 증액이 겹친 상황이다."""
    y1, y2 = make_weeks(seed, other_event)
    eff, m = seasonal_counterfactual_effect(y1, y2)
    return {
        "naive": naive_before_after(y2),
        "seasonal_cf": eff,
        "means": m,
        "true_effect": FEATURE_TRUE_EFFECT,
        "other_event": other_event,
        "season_rise": mean(season(w) for w in POST_WEEKS) - mean(season(w) for w in PRE_WEEKS),  # 노이즈 없는 계절 곡선만의 상승
    }


if __name__ == "__main__":
    a = scene_a()
    print("A 위젯: 관찰 격차 %.2f / 동전 배정 격차 %.2f / 진짜 %.2f (본 사람 %d명, 안 본 사람 %d명)" % (
        a["observed_gap"], a["randomized_gap"], a["true_effect"], a["n_seen_obs"], a["n_not_obs"]))
    for ev in (0.0, 12.0):
        b = scene_b(ev)
        print("B 기능, 광고 증액 %+.0f: 단순 전후 %.2f / 효과 어림값(작년 같은 주를 뺌) %.2f / 진짜 %.1f / 계절 곡선만의 상승 %.2f" % (
            ev, b["naive"], b["seasonal_cf"], b["true_effect"], b["season_rise"]))
        m = b["means"]
        print("  올해 전 %.2f 후 %.2f / 작년 전 %.2f 후 %.2f / 작년 같은 주 변화 %.2f" % (m["this_pre"], m["this_post"], m["last_pre"], m["last_post"], m["last_post"] - m["last_pre"]))
