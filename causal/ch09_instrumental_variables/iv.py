"""9장 도구변수. 표준 라이브러리만 쓴다. 가상 온라인 마켓의 장난감 데이터.

Z = 푸시 알림으로 쿠폰을 권유했는지(무작위), D = 쿠폰을 실제로 썼는지, Y = 한 달 지출(가상 원 단위 아님, 가상 '천원').
고객은 세 유형이다. 알림과 상관없이 늘 쿠폰을 쓰는 사람(always), 알림을 받아야 쓰는 사람(complier),
알림을 받아도 안 쓰는 사람(never). 도망치는 사람(defier)은 없다고 가정한다(단조성).

무엇을 재는가: 쿠폰을 쓴 사람과 안 쓴 사람의 단순 비교(선택 편향), 알림 배정 기준의 효과(ITT, Intent-To-Treat), 첫 단계(알림이 사용을 바꾼 정도),
Wald 추정량(= ITT / 첫 단계, 알림에 따라 움직이는 complier의 평균 효과인 LATE, Local Average Treatment Effect)을 구한다.
약한 도구(complier 비율이 낮음)에서 추정이 얼마나 흔들리는지, 배제제약 위반(알림이 쿠폰을 거치지 않고 지출을 올림)이 약한 도구에서 얼마나 증폭되는지 본다.
입력: simulate(n, seed, p_always, p_complier, direct, noise)의 행 (유형, z, d, y). 출력: 추정값 dict(report). 단위는 가상 '천원'.
테스트의 기대값(ITT 약 4.86, 첫 단계 약 0.503, Wald 약 10 등)이 장 본문 숫자와 대응한다.
핵심 가정: 관련성(Z가 D를 바꿈), 배제제약(Z는 D를 통해서만 Y에 영향), 단조성(defier 없음), Z의 무작위성.
"""
import random
from statistics import mean, pvariance, median

# 유형별 쿠폰 효과와 기본 지출(모두 가상)
EFFECT = {"always": 4.0, "complier": 10.0, "never": 0.0}
# 아래 BASE는 선택 편향의 원인이다. 자주 쓰는 사람이 원래 지출이 커서 쿠폰 사용자와 비사용자를 직접 비교하면 크게 부풀려진다
BASE = {"always": 60.0, "complier": 40.0, "never": 20.0}  # 자주 쓰는 사람이 원래 더 많이 산다 -> 선택 편향


def simulate(n=20000, seed=7, p_always=0.2, p_complier=0.5, direct=0.0, noise=8.0):
    """direct: 알림 자체가 쿠폰과 상관없이 지출에 주는 직접 효과(배제제약 위반의 크기).
    행은 (유형, z: 알림, d: 쿠폰 사용, y: 지출). 유형 비율은 p_always, p_complier, 나머지는 never."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        u = rng.random()  # 유형 추첨(누적 구간으로 나눈다)
        kind = "always" if u < p_always else ("complier" if u < p_always + p_complier else "never")
        z = 1 if rng.random() < 0.5 else 0  # 알림은 동전으로 무작위 배정
        # always는 z와 무관하게 사용, complier는 알림(z=1)일 때만 사용, never는 안 씀
        d = 1 if kind == "always" else (z if kind == "complier" else 0)
        # direct * z가 0이 아니면 알림이 d를 거치지 않고 y에 닿는다(배제제약 위반)
        y = BASE[kind] + EFFECT[kind] * d + direct * z + rng.gauss(0, noise)
        rows.append((kind, z, d, y))
    return rows


def group_mean(rows, idx, cond):
    """cond를 만족하는 행의 idx번째 열 평균(열 번호: 0 유형, 1 z, 2 d, 3 y)."""
    vals = [r[idx] for r in rows if cond(r)]
    return mean(vals)


def naive(rows):
    """쿠폰을 쓴 사람과 안 쓴 사람의 지출 평균 차이."""
    return group_mean(rows, 3, lambda r: r[2] == 1) - group_mean(rows, 3, lambda r: r[2] == 0)


def itt_y(rows):
    """알림 배정 기준 지출 차이(ITT). 알림을 받아도 쿠폰을 안 쓴 사람이 섞여 있어 효과가 희석된다."""
    return group_mean(rows, 3, lambda r: r[1] == 1) - group_mean(rows, 3, lambda r: r[1] == 0)


def first_stage(rows):
    """알림이 쿠폰 사용 비율을 바꾼 정도. 단조성이 맞으면 complier 비율과 같다(기본 설정에서 약 0.5)."""
    return group_mean(rows, 2, lambda r: r[1] == 1) - group_mean(rows, 2, lambda r: r[1] == 0)


def wald(rows):
    """Wald 추정량 = (알림이 지출을 바꾼 정도) / (알림이 쿠폰 사용을 바꾼 정도)."""
    return itt_y(rows) / first_stage(rows)


def first_stage_f(rows):
    """첫 단계 회귀(D = a + b Z)에서 b의 t값을 제곱한 F. 도구가 하나일 때 F = t^2."""
    n = len(rows)
    # 단순회귀 D = a + b Z의 기울기 b와 표준오차 se로 t = b / se, F = t^2
    z = [r[1] for r in rows]
    d = [r[2] for r in rows]
    zb, db = mean(z), mean(d)
    szz = sum((a - zb) ** 2 for a in z)
    szd = sum((a - zb) * (b - db) for a, b in zip(z, d))
    b = szd / szz
    resid = [(db_i - db) - b * (z_i - zb) for z_i, db_i in zip(z, d)]  # db_i는 d의 i번째 값(평균 db와 헷갈리지 않게)
    s2 = sum(e * e for e in resid) / (n - 2)  # 잔차 분산(모수 2개를 빼서 n - 2)
    se = (s2 / szz) ** 0.5
    return (b / se) ** 2


def true_late(rows):
    """장난감이라 유형을 아니까 진짜 LATE(complier의 효과)를 직접 낼 수 있다. rows는 쓰지 않는다."""
    return EFFECT["complier"]


def true_ate(rows):
    """진짜 ATE: 모든 유형의 효과를 유형 비율로 평균(약 5.8). LATE(10)와 다르다."""
    return mean(EFFECT[r[0]] for r in rows)


def repeated(reps=200, **kw):
    """같은 설정으로 표본을 reps번 새로 뽑아 Wald 추정값의 흩어짐을 본다."""
    est = []
    for k in range(reps):
        rows = simulate(seed=1000 + k, **kw)
        est.append(wald(rows))
    est.sort()
    # 분위수를 정렬 위치로 단순 추정한다(보간 없음). p10은 0부터 세어 int(0.10*reps)번째, p90은 int(0.90*reps) - 1번째라 위치 계산이 비대칭이다
    return {
        "median": median(est),
        "p10": est[int(0.10 * reps)],
        "p90": est[int(0.90 * reps) - 1],
        "sd": pvariance(est) ** 0.5,  # pvariance는 모분산(분모 n). 표본분산(n-1)이 아님
    }


def report():
    """테스트와 결과 표가 쓰는 숫자를 한 dict로 모은다(강한 도구 vs 약한 도구, F 통계량, 배제제약 위반 포함)."""
    rows = simulate()
    out = {
        "n": len(rows),
        "naive": naive(rows),
        "itt_y": itt_y(rows),
        "first_stage": first_stage(rows),
        "wald": wald(rows),
        "F": first_stage_f(rows),
        "true_late": true_late(rows),
        "true_ate": true_ate(rows),
    }
    # 강한 도구(complier 50%) vs 약한 도구(complier 3%): 같은 표본 크기에서 Wald 추정값의 흩어짐이 크게 달라진다
    strong = repeated(n=2000, p_always=0.2, p_complier=0.5)
    weak = repeated(n=2000, p_always=0.2, p_complier=0.03)
    out["strong_reps"] = strong
    out["weak_reps"] = weak
    w = simulate(n=20000, p_always=0.2, p_complier=0.03)
    out["weak_first_stage"] = first_stage(w)
    out["weak_F_n20000"] = first_stage_f(w)
    ws = simulate(n=2000, p_always=0.2, p_complier=0.03, seed=1000)
    out["weak_F_n2000"] = first_stage_f(ws)
    out["weak_F_mean_n2000"] = mean(first_stage_f(simulate(n=2000, seed=1000 + k, p_complier=0.03)) for k in range(200))
    out["weak_F_mean_n20000"] = mean(first_stage_f(simulate(n=20000, seed=1000 + k, p_complier=0.03)) for k in range(50))
    out["weak_wald_n20000_no_direct"] = wald(w)
    out["n_coupon_users"] = sum(r[2] for r in rows)
    out["n_notified"] = sum(r[1] for r in rows)
    for kind in ("always", "complier", "never"):
        out["share_" + kind] = sum(1 for r in rows if r[0] == kind) / len(rows)
    ss = simulate(n=2000, seed=1000)
    out["strong_F_n2000"] = first_stage_f(ss)
    # 배제제약 위반: 알림 자체가 지출을 +1(가상 천원) 올린다
    # Wald 편향 약 direct / 첫 단계: 강한 도구에서 +2(= 1 / 0.5), 약한 도구에서 약 +33(= 1 / 0.03)으로 증폭된다(근사)
    v_strong = simulate(n=20000, direct=1.0)
    v_weak = simulate(n=20000, p_complier=0.03, direct=1.0)
    out["violate_strong"] = wald(v_strong)
    out["violate_weak"] = wald(v_weak)
    return out


if __name__ == "__main__":
    for k, v in report().items():
        print(k, v if not isinstance(v, float) else round(v, 3))
