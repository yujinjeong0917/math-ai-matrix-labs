"""11장. 얼마를 쌓아 둘까. EOQ, 뉴스벤더(단일 시즌 발주), 안전재고. 모든 숫자는 학습용 가상값.

EOQ(Economic Order Quantity, 경제적 주문량) = 주문비와 보관비의 합이 최소인 1회 주문량.
뉴스벤더 = 시즌에 한 번만 발주하고 남으면 싸게 처분하는 단일 기간 재고 모형.
입력: 연 수요 2400개, 주문 1회 30,000원, 보관비 개당 연 1,200원, 시즌 수요 N(100, 30), 판매가·매입가·처분가, 주간 수요 N(100, 30)과 입고 2주.
대응(README 기대 출력): EOQ 346.41개·연 비용 415,692원(정수 탐색도 346), 임계비율 0.25와 뉴스벤더 79.77개(정수 탐색 80),
100개로 발주하면 80개보다 기대이익 약 77,906원 낮음, 안전재고 69.79·재주문점 269.79(95%), 시뮬레이션 달성률 0.950,
표준편차를 30으로 잘못 알고 실제 45이면 달성률 0.864.
"""
import math
import random
from statistics import NormalDist

# ---- EOQ ----------------------------------------------------------------
D = 2400          # 연간 수요(개)
K = 30_000        # 주문 1회 고정비(원)
H = 1_200         # 개당 연간 보관비(원)


def eoq_cost(q, d=D, k=K, h=H):
    """연간 주문비 + 보관비 = K*D/Q + h*Q/2 (구매 단가 항은 Q와 무관해 뺀다)."""
    # 주문비 = 연 주문 횟수(D/Q) x K, 보관비 = 평균 재고(Q/2) x h. 재고가 일정 속도로 줄어든다는 EOQ의 가정
    return k * d / q + h * q / 2


def eoq(d=D, k=K, h=H):
    # 비용식을 Q로 미분해 0으로 놓은 결과: Q* = sqrt(2DK/h) = sqrt(2 x 2400 x 30000 / 1200) = 346.41
    return math.sqrt(2 * d * k / h)


def eoq_by_scan(lo=1, hi=2000):
    # 공식 없이 정수 Q를 모두 대입해 비용 최소를 찾는 두 번째 길(대조용)
    return min(range(lo, hi + 1), key=eoq_cost)


def cost_ratio(q):
    """EOQ 비용 대비 Q로 주문할 때 비용 비율 = (Q*/Q + Q/Q*)/2."""
    # Q가 최적에서 벗어나도 비용은 완만히 늘어난다: Q=450이면 +3.4%, Q=100이면 +87.6%(test)
    qs = eoq()
    return (qs / q + q / qs) / 2


# ---- 뉴스벤더 ------------------------------------------------------------
MEAN, SD = 100, 30      # 시즌 총수요 평균·표준편차(개), 가상
PRICE, COST, SALVAGE = 36_000, 28_000, 4_000   # 판매가, 매입가, 남은 재고 처분가(원)


def critical_ratio(price=PRICE, cost=COST, salvage=SALVAGE):
    # Cu(underage) = 모자람 비용, Co(overage) = 남음 비용. 임계비율 Cu/(Cu+Co) = 8000/32000 = 0.25
    cu = price - cost        # 1개 모자랄 때 놓치는 이익
    co = cost - salvage      # 1개 남을 때 보는 손해
    return cu / (cu + co)


def demand_pmf(mean=MEAN, sd=SD, hi=300):
    """정수 수요 분포: 정규분포를 [d-0.5, d+0.5] 구간으로 쪼개고 음수 부분은 0에 합친다."""
    # 연속 정규분포를 정수 수요로 이산화(연속성 보정 +-0.5). 음수 수요 확률은 0개 쪽에, 300 이상은 마지막 칸에 합쳐 총합 1
    nd = NormalDist(mean, sd)
    pmf = [nd.cdf(0.5)] + [nd.cdf(d + 0.5) - nd.cdf(d - 0.5) for d in range(1, hi)]
    pmf.append(1 - sum(pmf))
    return pmf


def expected_profit(q, pmf=None, price=PRICE, cost=COST, salvage=SALVAGE):
    pmf = pmf or demand_pmf()
    total = 0.0
    for d, p in enumerate(pmf):
        # 팔린 수량 = min(발주량, 수요). 남은 (q - sold)개는 처분가로 판다. 매입비는 q개 전부에 든다
        sold = min(q, d)
        total += p * (price * sold + salvage * (q - sold) - cost * q)
    return total


def best_q_by_scan(pmf=None, **kw):
    pmf = pmf or demand_pmf()
    return max(range(0, 301), key=lambda q: expected_profit(q, pmf, **kw))


def newsvendor_q(mean=MEAN, sd=SD, **kw):
    # 최적 발주량 = 수요 분포의 임계비율 분위수(연속 근사). 정수 탐색 결과(80)와 반올림 수준에서 일치
    return NormalDist(mean, sd).inv_cdf(critical_ratio(**kw))


# ---- 안전재고 ------------------------------------------------------------
WEEK_MEAN, WEEK_SD, LEAD_WEEKS = 100, 30, 2


def safety_stock(service=0.95, sd_week=WEEK_SD, lead=LEAD_WEEKS):
    # z = 표준정규 분위수(95% -> 1.645). 입고 기간 L주 동안의 수요 표준편차는 sd_week x sqrt(L)(주간 수요가 서로 독립이라는 가정)
    z = NormalDist().inv_cdf(service)
    return z * sd_week * math.sqrt(lead)


def reorder_point(service=0.95, **kw):
    # 재주문점 = 입고 기간 평균 수요(100 x 2) + 안전재고
    return WEEK_MEAN * LEAD_WEEKS + safety_stock(service, **kw)


def simulated_cycle_service(rop, true_sd=WEEK_SD, cycles=100_000, seed=11):
    """재주문 후 입고까지(2주) 수요가 재주문점 이하인 주기의 비율."""
    # 시드 11 고정. 주기 단위 달성률("주기마다 품절이 없을 확률")이며 품절 개수 기준 서비스 수준과 다르다
    rng = random.Random(seed)
    ok = 0
    for _ in range(cycles):
        # 주간 수요가 음수가 되지 않도록 0으로 자른다(정규분포의 단순화에 대한 보정, 영향은 아주 작다)
        demand = sum(max(0.0, rng.gauss(WEEK_MEAN, true_sd)) for _ in range(LEAD_WEEKS))
        ok += demand <= rop
    return ok / cycles


def analytic_cycle_service(rop, true_sd=WEEK_SD):
    # 입고 기간 수요 ~ N(200, true_sd x sqrt(2))에서 재주문점 이하일 확률. true_sd를 틀리게 알면 달성률이 95%에서 86%로 떨어진다
    return NormalDist(WEEK_MEAN * LEAD_WEEKS, true_sd * math.sqrt(LEAD_WEEKS)).cdf(rop)


if __name__ == "__main__":
    print("EOQ", eoq(), "스캔", eoq_by_scan(), "비용", eoq_cost(eoq()))
    for q in (100, 200, 346, 450, 600, 1200):
        print(f"Q={q}: 연간 {eoq_cost(q):,.0f}원 비율 {cost_ratio(q):.3f}")
    print("주문 횟수/년", D / eoq(), "주기(일)", 365 * eoq() / D)
    print("임계비율", critical_ratio())
    pmf = demand_pmf()
    qn = newsvendor_q()
    print("뉴스벤더 공식 q", qn, "스캔 q", best_q_by_scan(pmf))
    for q in (100, 80, 79, 70, 60):
        print(f"q={q}: 기대이익 {expected_profit(q, pmf):,.0f}원")
    print("평균(100)대로 발주 vs 최적 이익 차", expected_profit(80, pmf) - expected_profit(100, pmf))
    print("품절 손실이 큰 경우 (매입가 28000, 판매가 60000)", critical_ratio(price=60_000), newsvendor_q(price=60_000))
    ss, rop = safety_stock(), reorder_point()
    print("안전재고", ss, "재주문점", rop)
    print("서비스 수준 90/95/99 안전재고", [round(safety_stock(s), 1) for s in (.9, .95, .99)])
    print("시뮬 서비스", simulated_cycle_service(rop), "해석", analytic_cycle_service(rop))
    print("표준편차를 30으로 잘못 알았는데 실제 45", analytic_cycle_service(rop, 45), simulated_cycle_service(rop, 45))
