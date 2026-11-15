"""10장. 이용률과 대기. 상담원 1명이 문의를 선착순으로 처리하는 M/M/1 모형(가상 숫자).
M/M/1 = 도착 간격과 처리 시간이 모두 지수분포(무기억), 상담원 1명, 선착순.

재는 것: 이용률 rho(처리 능력 중 바쁜 비율)가 올라갈 때 평균 체류 시간과 긴 대기가 얼마나 비선형으로 늘어나는지.
입력: rho(0~1 미만), 평균 처리 시간 5분(가상). 출력: 평균 체류 W, 평균 대기 Wq, 분위수, 30분 초과 비율, 시뮬레이션 값.
대응(README 기대 출력): rho 0.5, 0.7, 0.8, 0.9, 0.95에서 W = 10.0, 16.7, 25.0, 50.0, 100.0분,
rho 0.8의 30분 초과 비율 공식 0.301 대 시뮬레이션 0.310, 평균 체류 30분을 맞추는 rho 0.833에서도 초과 비율 0.368.
M/M/1은 정상상태(오래 돌린 평균)의 식이다.
"""
import math
import random

MEAN_SERVICE = 5.0           # 문의 1건 평균 처리 시간(분), 가상
MU = 1 / MEAN_SERVICE        # 처리율(건/분)


def lam_for(rho):
    # 이용률 rho = 도착률 lambda / 처리율 mu 이므로 lambda = rho * mu (건/분)
    return rho * MU


def mm1_mean_system(rho):
    """접수부터 답변 끝까지 평균 시간 W = 1/(mu-lambda) (분)."""
    # rho -> 1이면 분모가 0에 가까워 급증: 5/(1-rho)분. 0.9에서 50분, 0.95에서 100분
    return 1 / (MU - lam_for(rho))


def mm1_mean_wait(rho):
    """줄에서 기다리는 평균 시간 Wq = rho/(mu-lambda) (분)."""
    return rho / (MU - lam_for(rho))


def mm1_tail(rho, t):
    """P(접수부터 답변 끝까지 > t분) = exp(-(mu-lambda)t)."""
    # M/M/1에서 체류 시간은 평균 W인 지수분포(선착순일 때만 성립)
    return math.exp(-(MU - lam_for(rho)) * t)


def mm1_quantile(rho, q):
    """접수부터 답변 끝까지 시간의 q 분위수(분)."""
    # 위 꼬리식을 역으로 푼 값: P(T > t) = 1 - q 가 되는 t
    return -math.log(1 - q) / (MU - lam_for(rho))


def md1_mean_wait(rho):
    """처리 시간이 항상 같을 때(M/D/1) 평균 대기 = lambda*E[S^2]/(2(1-rho)) (Pollaczek-Khinchine)."""
    # 처리 시간이 상수 5분이면 E[S^2] = 25. 같은 rho에서 M/M/1(E[S^2] = 2 x 25)의 정확히 절반
    lam = lam_for(rho)
    return lam * MEAN_SERVICE ** 2 / (2 * (1 - rho))


def simulate(rho, n=200_000, seed=7, deterministic_service=False):
    """Lindley 점화식 Wq[k+1] = max(0, Wq[k] + S[k] - A[k+1])로 n건을 시뮬레이션한다.
    (평균 대기, 평균 체류, 체류시간 목록) 반환."""
    # 시드 7 고정: 같은 숫자가 다시 나와 테스트의 기대값(20.51 등)이 재현된다
    rng = random.Random(seed)
    lam = lam_for(rho)
    wq, tot_wq, times = 0.0, 0.0, []
    for _ in range(n):
        # 이 고객의 체류 = 기다린 시간 + 처리 시간. 상수 처리 시간 모드에서는 난수를 덜 뽑아 난수열이 달라진다
        s = MEAN_SERVICE if deterministic_service else rng.expovariate(MU)
        times.append(wq + s)
        tot_wq += wq
        # 린들리 점화식: 다음 고객의 대기 = 지금 대기 + 내 처리 시간 - 다음 도착까지 간격(음수면 상담원이 놀았으므로 0)
        wq = max(0.0, wq + s - rng.expovariate(lam))
    return tot_wq / n, sum(times) / n, times


def share_over(times, t):
    return sum(1 for x in times if x > t) / len(times)


def quantile(times, q):
    # q 분위수 = 오름차순 정렬의 (q*n)번째 값(0부터 세므로 -1). 보간하지 않는 단순 정의
    xs = sorted(times)
    return xs[int(q * len(xs)) - 1]


if __name__ == "__main__":
    print("rho  W(분)  Wq(분)  p95(분)  P(>30분)")
    for rho in (0.5, 0.7, 0.8, 0.9, 0.95):
        print(f"{rho:.2f} {mm1_mean_system(rho):6.1f} {mm1_mean_wait(rho):6.1f} {mm1_quantile(rho, .95):7.1f} {mm1_tail(rho, 30):8.3f}")
    print("--- 시뮬레이션 (seed=7, 20만 건)")
    for rho in (0.5, 0.8, 0.9):
        wq, w, ts = simulate(rho)
        print(f"rho={rho}: Wq={wq:.2f} W={w:.2f} p95={quantile(ts, .95):.1f} P(>30)={share_over(ts, 30):.3f}")
    print("--- 처리 시간이 항상 5분(M/D/1), rho=0.8")
    print("공식 Wq", md1_mean_wait(0.8), "M/M/1 Wq", mm1_mean_wait(0.8))
    wq, w, ts = simulate(0.8, deterministic_service=True)
    print(f"시뮬레이션 Wq={wq:.2f}")
    # 0.9 x 1.08 = 0.972: 요청이 8%만 늘어도 평균 체류가 50분에서 약 178.6분으로 뛴다
    print("--- 요청이 8% 늘어날 때(rho 0.9 -> 0.972)")
    print(mm1_mean_system(0.9), mm1_mean_system(0.972))
    print("--- 평균 W<=30분 만족하는 최대 rho와 그때 P(>30)")
    # W = 5/(1-rho) = 30 을 rho로 풀면 1 - 5/30 = 0.8333. 평균은 맞아도 체류가 30분을 넘는 고객이 e^-1 = 36.8%
    rho = 1 - MEAN_SERVICE / 30
    print(rho, mm1_tail(rho, 30))
