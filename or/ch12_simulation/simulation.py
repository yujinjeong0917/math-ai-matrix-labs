"""12장. 이산사건 시뮬레이션. 상담 센터(상담원 c명, 선착순, 한 줄 대기)를 사건 목록(heapq)으로 돌린다.
숫자는 모두 학습용 가상값. 난수는 random.Random(시드)만 쓴다.

재는 것: 상담원 수(c)에 따른 평균 대기 시간. 한 번의 실행은 난수에 크게 흔들리므로 시드를 바꿔 20회 반복해 평균과 95% 신뢰구간을 낸다.
입력: 근무 480분, 처음 120분은 도착률 0.36건/분, 이후 0.16건/분, 처리 시간은 평균 5분 로그정규분포. 출력: 고객별 대기 시간(분).
대응(README 기대 출력): M/M/1 검증 시뮬레이션 19.71분 대 이론 20.00분, 상담원 2명 한 번 실행은 시드별 0.62~13.98분,
20회 평균 3.77분(2.20~5.34), 3명이면 0.58분, 같은 난수로 짝지어 뺀 차이 3.19분(1.88~4.49), 난수를 따로 쓰면 3.38분(1.78~4.97).
"""
import heapq
import math
import random
from statistics import mean, stdev

# 반복 횟수가 20이 아니면 이 값을 쓸 수 없다(ci95가 길이 20을 assert하는 이유)
T_CRIT_19DF = 2.093   # 반복 20회, 양쪽 95% t 임계값(자유도 19). 위키백과 t분포 표에서 확인
SHIFT = 480           # 근무 시간(분)
PEAK_END = 120        # 처음 2시간은 바쁜 시간대
LAM_PEAK, LAM_BASE = 0.36, 0.16   # 분당 도착률(건/분), 가상
LAM_MAX = LAM_PEAK   # 솎아내기에서 쓰는 도착률 상한(어느 시간대든 이 값 이하여야 한다)
SERVICE_MEAN = 5.0
LOG_SIGMA = 0.8       # 처리 시간: 평균 5분인 로그정규분포


def lam_at(t):
    return LAM_PEAK if t < PEAK_END else LAM_BASE


def make_arrivals(seed):
    """시간대별 도착률을 솎아내기(thinning)로 만든 도착 시각 목록."""
    # 솎아내기(thinning): 가장 높은 도착률로 후보 시각을 만든 뒤 lam(t)/LAM_MAX 확률로만 남겨 시간대별 도착률을 재현
    rng = random.Random(seed)
    t, out = 0.0, []
    while True:
        t += rng.expovariate(LAM_MAX)
        if t >= SHIFT:
            return out
        if rng.random() < lam_at(t) / LAM_MAX:
            out.append(t)


def make_services(seed, n):
    # 도착과 다른 난수열(시드 변환)을 써서 도착 난수와 섞이지 않게 한다. 같은 시드면 상담원 수가 달라도 같은 도착·처리 시간(공통 난수)
    rng = random.Random(seed * 1000 + 1)
    # 로그정규의 평균 = exp(mu + sigma^2/2) 이므로 mu를 이렇게 잡으면 평균이 5분
    mu = math.log(SERVICE_MEAN) - LOG_SIGMA ** 2 / 2   # 평균이 SERVICE_MEAN이 되게
    return [rng.lognormvariate(mu, LOG_SIGMA) for _ in range(n)]


def run(arrivals, services, c):
    """사건 목록으로 선착순 다중 상담원 시스템을 돌려 고객별 대기 시간을 돌려준다.
    근무 종료 후 도착한 고객은 없고, 이미 들어온 고객은 끝까지 처리한다."""
    events = [(a, 0, i) for i, a in enumerate(arrivals)]   # (시각, 종류 0=도착 1=종료, 번호)
    # 도착 사건을 시각순 힙에 넣는다. 종료 사건은 서비스를 시작할 때 추가된다. 같은 시각이면 종류(0 도착, 1 종료) 순
    heapq.heapify(events)
    busy, queue, waits = 0, [], [0.0] * len(arrivals)
    while events:
        t, kind, i = heapq.heappop(events)
        if kind == 0:
            queue.append(i)
        else:
            busy -= 1
        # 빈 상담원이 있고 기다리는 사람이 있으면 선착순으로 시작. 대기 = 시작 시각 - 도착 시각. 처리 시간은 고객 번호 j에 붙어 있다
        while busy < c and queue:
            j = queue.pop(0)
            waits[j] = t - arrivals[j]
            busy += 1
            heapq.heappush(events, (t + services[j], 1, j))
    return waits


def one_replication(seed, c):
    arr = make_arrivals(seed)
    return run(arr, make_services(seed, len(arr)), c)


def summarize(waits, limit=10):
    return mean(waits), sum(1 for w in waits if w > limit) / len(waits)


def replicate(c, seeds, stat=0):
    return [summarize(one_replication(s, c))[stat] for s in seeds]


def ci95(xs):
    """반복 20회 전용 95% 신뢰구간(평균 +- t*s/sqrt(n))."""
    # 95% 구간 = 평균 +- t(자유도 19) x 표준오차. 값이 한쪽으로 치우친 통계량(비율 등)에는 근사일 뿐이다
    assert len(xs) == 20
    m, s = mean(xs), stdev(xs)
    half = T_CRIT_19DF * s / math.sqrt(len(xs))
    return m, m - half, m + half


def mm1_check(rho=0.8, n=200_000, seed=3):
    """같은 사건 엔진(c=1)을 M/M/1 조건에 돌려 이론값 Wq=rho*5/(1-rho)와 비교."""
    # 엔진 검증: 이론값을 아는 M/M/1(도착·처리 모두 지수분포)에 같은 run()을 돌려 맞는지 본다
    rng = random.Random(seed)
    lam, mu = rho / SERVICE_MEAN, 1 / SERVICE_MEAN
    t, arr = 0.0, []
    for _ in range(n):
        t += rng.expovariate(lam)
        arr.append(t)
    sv = [rng.expovariate(mu) for _ in range(n)]
    return mean(run(arr, sv, 1)), rho * SERVICE_MEAN / (1 - rho)


if __name__ == "__main__":
    print("엔진 검증 M/M/1 rho=0.8 (시뮬, 이론):", mm1_check())
    print("--- 상담원 2명, 시드별 한 번 실행한 평균 대기(분)")
    for s in range(1, 6):
        w = one_replication(s, 2)
        print(s, len(w), [round(x, 2) for x in summarize(w)])
    seeds = list(range(1, 21))
    for c in (2, 3):
        xs = replicate(c, seeds)
        print(f"c={c} 평균대기 20회:", [round(x, 2) for x in xs])
        print("  평균, 95% 구간:", [round(v, 2) for v in ci95(xs)])
        ys = replicate(c, seeds, stat=1)
        print("  10분 초과 비율 평균, 구간:", [round(v, 3) for v in ci95(ys)])
        print("  10분 초과 비율 20회:", [round(v, 3) for v in ys])
    # 같은 시드끼리 짝지어 빼면 난수로 인한 흔들림이 상쇄되어 구간이 좁다(1.88~4.49 대 1.78~4.97)
    print("--- 같은 난수로 짝지어 뺀 차이(공통 난수) c=2 - c=3")
    d = [a - b for a, b in zip(replicate(2, seeds), replicate(3, seeds))]
    print([round(v, 2) for v in ci95(d)])
    print("--- 다른 난수로 따로 돌렸을 때(c=3은 시드 101~120)")
    d2 = [a - b for a, b in zip(replicate(2, seeds), replicate(3, range(101, 121)))]
    print([round(v, 2) for v in ci95(d2)])
    # 하루 평균(3.77)에 가려진 바쁜 시간대의 대기: 도착 시각이 120분 전인 고객만 평균낸다
    print("--- 처음 2시간만 본 값(바쁜 시간대) vs 전체")
    pk = []
    for s in seeds:
        arr = make_arrivals(s); w = run(arr, make_services(s, len(arr)), 2)
        pk.append(mean([x for a, x in zip(arr, w) if a < PEAK_END]))
    print(round(mean(pk), 2))
