"""11장. 얼마를 쌓아 둘까. EOQ, 뉴스벤더(단일 시즌 발주), 안전재고. 모든 숫자는 학습용 가상값."""
import math
import random
from statistics import NormalDist

# ---- EOQ ----------------------------------------------------------------
D = 2400          # 연간 수요(개)
K = 30_000        # 주문 1회 고정비(원)
H = 1_200         # 개당 연간 보관비(원)


def eoq_cost(q, d=D, k=K, h=H):
    """연간 주문비 + 보관비 = K*D/Q + h*Q/2 (구매 단가 항은 Q와 무관해 뺀다)."""
    return k * d / q + h * q / 2


def eoq(d=D, k=K, h=H):
    return math.sqrt(2 * d * k / h)


def eoq_by_scan(lo=1, hi=2000):
    return min(range(lo, hi + 1), key=eoq_cost)


def cost_ratio(q):
    """EOQ 비용 대비 Q로 주문할 때 비용 비율 = (Q*/Q + Q/Q*)/2."""
    qs = eoq()
    return (qs / q + q / qs) / 2


# ---- 뉴스벤더 ------------------------------------------------------------
MEAN, SD = 100, 30      # 시즌 총수요 평균·표준편차(개), 가상
PRICE, COST, SALVAGE = 36_000, 28_000, 4_000   # 판매가, 매입가, 남은 재고 처분가(원)


def critical_ratio(price=PRICE, cost=COST, salvage=SALVAGE):
    cu = price - cost        # 1개 모자랄 때 놓치는 이익
    co = cost - salvage      # 1개 남을 때 보는 손해
    return cu / (cu + co)


def demand_pmf(mean=MEAN, sd=SD, hi=300):
    """정수 수요 분포: 정규분포를 [d-0.5, d+0.5] 구간으로 쪼개고 음수 부분은 0에 합친다."""
    nd = NormalDist(mean, sd)
    pmf = [nd.cdf(0.5)] + [nd.cdf(d + 0.5) - nd.cdf(d - 0.5) for d in range(1, hi)]
    pmf.append(1 - sum(pmf))
    return pmf


def expected_profit(q, pmf=None, price=PRICE, cost=COST, salvage=SALVAGE):
    pmf = pmf or demand_pmf()
    total = 0.0
    for d, p in enumerate(pmf):
        sold = min(q, d)
        total += p * (price * sold + salvage * (q - sold) - cost * q)
    return total


def best_q_by_scan(pmf=None, **kw):
    pmf = pmf or demand_pmf()
    return max(range(0, 301), key=lambda q: expected_profit(q, pmf, **kw))


def newsvendor_q(mean=MEAN, sd=SD, **kw):
    return NormalDist(mean, sd).inv_cdf(critical_ratio(**kw))


# ---- 안전재고 ------------------------------------------------------------
WEEK_MEAN, WEEK_SD, LEAD_WEEKS = 100, 30, 2


def safety_stock(service=0.95, sd_week=WEEK_SD, lead=LEAD_WEEKS):
    z = NormalDist().inv_cdf(service)
    return z * sd_week * math.sqrt(lead)


def reorder_point(service=0.95, **kw):
    return WEEK_MEAN * LEAD_WEEKS + safety_stock(service, **kw)


def simulated_cycle_service(rop, true_sd=WEEK_SD, cycles=100_000, seed=11):
    """재주문 후 입고까지(2주) 수요가 재주문점 이하인 주기의 비율."""
    rng = random.Random(seed)
    ok = 0
    for _ in range(cycles):
        demand = sum(max(0.0, rng.gauss(WEEK_MEAN, true_sd)) for _ in range(LEAD_WEEKS))
        ok += demand <= rop
    return ok / cycles


def analytic_cycle_service(rop, true_sd=WEEK_SD):
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
