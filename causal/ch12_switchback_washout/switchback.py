"""12장 스위치백과 세척기간. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 지역 한 곳의 '오후 타임세일 문구'를 시간대마다 켰다 껐다 한다.

Y_t = 기본_t + tau*T_t + k1*T_{t-1} + k2*T_{t-2} + 잡음.
tau=+6은 문구를 켠 바로 그 시간대의 효과, k1=-3, k2=-1.5는 '앞당겨 산 뒤 줄어드는' 이월효과(가상).
계속 켜 둘 때(항상 켬)와 계속 꺼 둘 때(항상 끔)의 정상상태 차이 = tau+k1+k2 = 1.5.
"""
import random
from statistics import mean

TAU, K1, K2 = 6.0, -3.0, -1.5
TRUE_STEADY = TAU + K1 + K2
N = 240  # 시간대 슬롯 수(가상 2시간 단위로 20일)


def base_curve(t):
    return 100 + 10 * (1 if (t % 12) in (4, 5, 6, 7) else 0) + 4 * ((t // 12) % 7 in (5, 6))


def assign(block, rng, n=N):
    """블록 길이 block마다 동전을 던져 켬/끔을 정한다(블록 수 n/block, 켬 절반 맞춤)."""
    nb = n // block
    flags = [1] * (nb // 2) + [0] * (nb - nb // 2)
    rng.shuffle(flags)
    return [flags[t // block] for t in range(n)]


def outcomes(T, rng, noise=3.0, flat=False):
    """flat=True면 시간대·요일 패턴을 없앤(기본 100 고정) 대조용 데이터."""
    y = []
    for t in range(len(T)):
        l1 = T[t - 1] if t >= 1 else 0
        l2 = T[t - 2] if t >= 2 else 0
        y.append((100 if flat else base_curve(t)) + TAU * T[t] + K1 * l1 + K2 * l2 + rng.gauss(0, noise))
    return y


def estimate(T, y, washout):
    """블록이 바뀐 뒤 washout개 슬롯은 버리고 켬 평균 - 끔 평균. 블록 길이는 T에서 읽는다."""
    keep = []
    run = 0
    prev = None
    for t, v in enumerate(T):
        run = run + 1 if v == prev else 1
        prev = v
        if run > washout:
            keep.append(t)
    a = [y[t] for t in keep if T[t] == 1]  # 기본 곡선(시간대·요일 패턴)은 분석자가 모른다고 보고 빼지 않는다
    b = [y[t] for t in keep if T[t] == 0]
    return mean(a) - mean(b)


def kept_share(block, washout, reps=300):
    """세척기간을 버리고 남는 슬롯 비율의 평균(배정 reps번).
    상태가 실제로 바뀐 뒤에만 버리고, 같은 상태가 이어지면 블록이 합쳐져 버리지 않는다.
    손계산: 1 - (washout/block) x (상태가 바뀔 확률 약 1/2)."""
    shares = []
    for r in range(reps):
        T = assign(block, random.Random(r))
        run, prev, k = 0, None, 0
        for v in T:
            run = run + 1 if v == prev else 1
            prev = v
            k += run > washout
        shares.append(k / len(T))
    return mean(shares)


def repeated(block, washout, reps=300, seed=0, flat=False):
    est = []
    for r in range(reps):
        rng = random.Random(seed * 100000 + r)
        T = assign(block, rng)
        y = outcomes(T, rng, flat=flat)
        est.append(estimate(T, y, washout))
    m = mean(est)
    sd = mean([(x - m) ** 2 for x in est]) ** 0.5
    return {"mean": m, "sd": sd, "bias": m - TRUE_STEADY}


def report():
    rows = {}
    for block, w in ((1, 0), (2, 0), (4, 0), (4, 1), (4, 2), (8, 2), (12, 2), (24, 2), (60, 2)):
        rows[(block, w)] = repeated(block, w)
    return rows


def flat_sd_report():
    """시간대·요일 패턴이 없을 때의 표준편차(잡음과 배정만의 흩어짐)."""
    return {(b, w): repeated(b, w, flat=True)["sd"] for b, w in ((1, 0), (4, 2), (12, 2), (24, 2), (60, 2))}


def kept_report():
    return {(b, w): kept_share(b, w) for b, w in ((4, 1), (4, 2), (8, 2), (24, 2), (60, 2))}


if __name__ == "__main__":
    print("true steady", TRUE_STEADY)
    for (b, w), v in report().items():
        print(f"block={b:>2} washout={w}", {k: round(x, 2) for k, x in v.items()})
    for (b, w), v in kept_report().items():
        print(f"kept_share block={b:>2} washout={w}", round(v, 3))
    for (b, w), v in flat_sd_report().items():
        print(f"flat_base_sd block={b:>2} washout={w}", round(v, 2))
