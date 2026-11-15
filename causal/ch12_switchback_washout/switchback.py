"""12장 스위치백과 세척기간. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 지역 한 곳의 '오후 타임세일 문구'를 시간대마다 켰다 껐다 한다.

Y_t = 기본_t + tau*T_t + k1*T_{t-1} + k2*T_{t-2} + 잡음.
tau=+6은 문구를 켠 바로 그 시간대의 효과, k1=-3, k2=-1.5는 '앞당겨 산 뒤 줄어드는' 이월효과(가상).
계속 켜 둘 때(항상 켬)와 계속 꺼 둘 때(항상 끔)의 정상상태 차이 = tau+k1+k2 = 1.5.

무엇을 재는가: 켬/끔을 블록(연속 슬롯 묶음)별로 무작위로 번갈아 하는 스위치백(switchback) 실험에서, 블록 길이와
세척기간(washout, 상태가 바뀐 직후 이월효과가 섞이는 슬롯을 분석에서 버리는 길이)이 "정상상태 효과" 추정의 편향·흩어짐에 주는 영향을 본다.
입력: 슬롯 수 N = 240(2시간 단위 20일), 블록 길이 block, 세척기간 washout. 출력: 추정값 평균·표준편차·편향(repeated), 남는 슬롯 비율(kept_share).
테스트의 기대값(정상상태 1.5, 1시간 단위 교대 추정 약 6, 남는 비율 0.75 등)이다(장 본문 숫자와의 대응은 README로 확인 필요). 효과 단위는 가상의 주문 수.
이월효과가 2슬롯까지만 있다는 것은 이 모형의 가정이다(현실에서는 이월 길이를 모른다).
"""
import random
from statistics import mean

TAU, K1, K2 = 6.0, -3.0, -1.5
TRUE_STEADY = TAU + K1 + K2
N = 240  # 시간대 슬롯 수(가상 2시간 단위로 20일)


def base_curve(t):
    """분석자가 모르는 기본 수요: 하루 12슬롯 중 4~7번째 슬롯이 +10(피크 시간대), 7일 주기 중 마지막 두 날(5, 6)이 +4(주말에 해당한다고 둔 모형의 선택)."""
    return 100 + 10 * (1 if (t % 12) in (4, 5, 6, 7) else 0) + 4 * ((t // 12) % 7 in (5, 6))


def assign(block, rng, n=N):
    """블록 길이 block마다 동전을 던져 켬/끔을 정한다(블록 수 n/block, 켬 절반 맞춤)."""
    nb = n // block  # 블록 수. n이 block의 배수일 때만 정확하다(N = 240은 1, 2, 4, 8, 12, 24, 60의 배수)
    flags = [1] * (nb // 2) + [0] * (nb - nb // 2)
    rng.shuffle(flags)  # 켬/끔 개수는 고정(켬 nb // 2개)하고 순서만 무작위로
    return [flags[t // block] for t in range(n)]


def outcomes(T, rng, noise=3.0, flat=False):
    """flat=True면 시간대·요일 패턴을 없앤(기본 100 고정) 대조용 데이터."""
    y = []
    for t in range(len(T)):
        # t-1, t-2의 처치를 이월효과로 반영. 실험 시작 전(t < 1, 2)은 "꺼져 있었다"고 둔다
        l1 = T[t - 1] if t >= 1 else 0
        l2 = T[t - 2] if t >= 2 else 0
        y.append((100 if flat else base_curve(t)) + TAU * T[t] + K1 * l1 + K2 * l2 + rng.gauss(0, noise))
    return y


def estimate(T, y, washout):
    """블록이 바뀐 뒤 washout개 슬롯은 버리고 켬 평균 - 끔 평균. 블록 길이는 T에서 읽는다."""
    keep = []
    run = 0   # 현재 상태가 연속된 슬롯 수
    prev = None
    for t, v in enumerate(T):
        # 같은 상태가 이어지면 run이 계속 늘고(블록이 합쳐져 버리지 않는다), 상태가 바뀌면 1로 되돌아간다
        run = run + 1 if v == prev else 1
        prev = v
        if run > washout:  # 바뀐 직후 washout개 슬롯은 이전 상태의 이월이 섞여 있어 제외
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
        run, prev, k = 0, None, 0  # k: 남는 슬롯 수(위 estimate와 같은 규칙)
        for v in T:
            run = run + 1 if v == prev else 1
            prev = v
            k += run > washout
        shares.append(k / len(T))
    return mean(shares)


def repeated(block, washout, reps=300, seed=0, flat=False):
    """배정과 잡음을 reps번 새로 뽑아 추정값의 평균, 표준편차(모표준편차), 정상상태 대비 편향을 구한다."""
    est = []
    for r in range(reps):
        rng = random.Random(seed * 100000 + r)
        T = assign(block, rng)  # 같은 rng로 배정 후 잡음을 이어 뽑는다(순서를 바꾸면 모든 숫자가 달라진다)
        y = outcomes(T, rng, flat=flat)
        est.append(estimate(T, y, washout))
    m = mean(est)
    sd = mean([(x - m) ** 2 for x in est]) ** 0.5
    return {"mean": m, "sd": sd, "bias": m - TRUE_STEADY}


def report():
    """(블록 길이, 세척기간) 조합별 repeated 결과. 키는 (block, washout) 튜플."""
    rows = {}
    for block, w in ((1, 0), (2, 0), (4, 0), (4, 1), (4, 2), (8, 2), (12, 2), (24, 2), (60, 2)):
        rows[(block, w)] = repeated(block, w)
    return rows


def flat_sd_report():
    """시간대·요일 패턴이 없을 때의 표준편차(잡음과 배정만의 흩어짐)."""
    return {(b, w): repeated(b, w, flat=True)["sd"] for b, w in ((1, 0), (4, 2), (12, 2), (24, 2), (60, 2))}


def kept_report():
    """조합별 남는 슬롯 비율. 세척기간을 길게 잡을수록, 블록이 짧을수록 데이터를 더 버린다."""
    return {(b, w): kept_share(b, w) for b, w in ((4, 1), (4, 2), (8, 2), (24, 2), (60, 2))}


if __name__ == "__main__":
    print("true steady", TRUE_STEADY)
    for (b, w), v in report().items():
        print(f"block={b:>2} washout={w}", {k: round(x, 2) for k, x in v.items()})
    for (b, w), v in kept_report().items():
        print(f"kept_share block={b:>2} washout={w}", round(v, 3))
    for (b, w), v in flat_sd_report().items():
        print(f"flat_base_sd block={b:>2} washout={w}", round(v, 2))
