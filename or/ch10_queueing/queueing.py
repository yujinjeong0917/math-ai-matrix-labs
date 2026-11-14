"""10장. 이용률과 대기. 상담원 1명이 문의를 선착순으로 처리하는 M/M/1 모형(가상 숫자).
M/M/1 = 도착 간격과 처리 시간이 모두 지수분포(무기억), 상담원 1명, 선착순."""
import math
import random

MEAN_SERVICE = 5.0           # 문의 1건 평균 처리 시간(분), 가상
MU = 1 / MEAN_SERVICE        # 처리율(건/분)


def lam_for(rho):
    return rho * MU


def mm1_mean_system(rho):
    """접수부터 답변 끝까지 평균 시간 W = 1/(mu-lambda) (분)."""
    return 1 / (MU - lam_for(rho))


def mm1_mean_wait(rho):
    """줄에서 기다리는 평균 시간 Wq = rho/(mu-lambda) (분)."""
    return rho / (MU - lam_for(rho))


def mm1_tail(rho, t):
    """P(접수부터 답변 끝까지 > t분) = exp(-(mu-lambda)t)."""
    return math.exp(-(MU - lam_for(rho)) * t)


def mm1_quantile(rho, q):
    """접수부터 답변 끝까지 시간의 q 분위수(분)."""
    return -math.log(1 - q) / (MU - lam_for(rho))


def md1_mean_wait(rho):
    """처리 시간이 항상 같을 때(M/D/1) 평균 대기 = lambda*E[S^2]/(2(1-rho)) (Pollaczek-Khinchine)."""
    lam = lam_for(rho)
    return lam * MEAN_SERVICE ** 2 / (2 * (1 - rho))


def simulate(rho, n=200_000, seed=7, deterministic_service=False):
    """Lindley 점화식 Wq[k+1] = max(0, Wq[k] + S[k] - A[k+1])로 n건을 시뮬레이션한다.
    (평균 대기, 평균 체류, 체류시간 목록) 반환."""
    rng = random.Random(seed)
    lam = lam_for(rho)
    wq, tot_wq, times = 0.0, 0.0, []
    for _ in range(n):
        s = MEAN_SERVICE if deterministic_service else rng.expovariate(MU)
        times.append(wq + s)
        tot_wq += wq
        wq = max(0.0, wq + s - rng.expovariate(lam))
    return tot_wq / n, sum(times) / n, times


def share_over(times, t):
    return sum(1 for x in times if x > t) / len(times)


def quantile(times, q):
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
    print("--- 요청이 8% 늘어날 때(rho 0.9 -> 0.972)")
    print(mm1_mean_system(0.9), mm1_mean_system(0.972))
    print("--- 평균 W<=30분 만족하는 최대 rho와 그때 P(>30)")
    rho = 1 - MEAN_SERVICE / 30
    print(rho, mm1_tail(rho, 30))
