"""9장 도구변수. 표준 라이브러리만 쓴다. 가상 온라인 마켓의 장난감 데이터.

Z = 푸시 알림으로 쿠폰을 권유했는지(무작위), D = 쿠폰을 실제로 썼는지, Y = 한 달 지출(가상 원 단위 아님, 가상 '천원').
고객은 세 유형이다. 알림과 상관없이 늘 쿠폰을 쓰는 사람(always), 알림을 받아야 쓰는 사람(complier),
알림을 받아도 안 쓰는 사람(never). 도망치는 사람(defier)은 없다고 가정한다(단조성).
"""
import random
from statistics import mean, pvariance, median

# 유형별 쿠폰 효과와 기본 지출(모두 가상)
EFFECT = {"always": 4.0, "complier": 10.0, "never": 0.0}
BASE = {"always": 60.0, "complier": 40.0, "never": 20.0}  # 자주 쓰는 사람이 원래 더 많이 산다 -> 선택 편향


def simulate(n=20000, seed=7, p_always=0.2, p_complier=0.5, direct=0.0, noise=8.0):
    """direct: 알림 자체가 쿠폰과 상관없이 지출에 주는 직접 효과(배제제약 위반의 크기)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        u = rng.random()
        kind = "always" if u < p_always else ("complier" if u < p_always + p_complier else "never")
        z = 1 if rng.random() < 0.5 else 0
        d = 1 if kind == "always" else (z if kind == "complier" else 0)
        y = BASE[kind] + EFFECT[kind] * d + direct * z + rng.gauss(0, noise)
        rows.append((kind, z, d, y))
    return rows


def group_mean(rows, idx, cond):
    vals = [r[idx] for r in rows if cond(r)]
    return mean(vals)


def naive(rows):
    """쿠폰을 쓴 사람과 안 쓴 사람의 지출 평균 차이."""
    return group_mean(rows, 3, lambda r: r[2] == 1) - group_mean(rows, 3, lambda r: r[2] == 0)


def itt_y(rows):
    return group_mean(rows, 3, lambda r: r[1] == 1) - group_mean(rows, 3, lambda r: r[1] == 0)


def first_stage(rows):
    return group_mean(rows, 2, lambda r: r[1] == 1) - group_mean(rows, 2, lambda r: r[1] == 0)


def wald(rows):
    """Wald 추정량 = (알림이 지출을 바꾼 정도) / (알림이 쿠폰 사용을 바꾼 정도)."""
    return itt_y(rows) / first_stage(rows)


def first_stage_f(rows):
    """첫 단계 회귀(D = a + b Z)에서 b의 t값을 제곱한 F. 도구가 하나일 때 F = t^2."""
    n = len(rows)
    z = [r[1] for r in rows]
    d = [r[2] for r in rows]
    zb, db = mean(z), mean(d)
    szz = sum((a - zb) ** 2 for a in z)
    szd = sum((a - zb) * (b - db) for a, b in zip(z, d))
    b = szd / szz
    resid = [(db_i - db) - b * (z_i - zb) for z_i, db_i in zip(z, d)]
    s2 = sum(e * e for e in resid) / (n - 2)
    se = (s2 / szz) ** 0.5
    return (b / se) ** 2


def true_late(rows):
    """장난감이라 유형을 아니까 진짜 값을 직접 낼 수 있다."""
    return EFFECT["complier"]


def true_ate(rows):
    return mean(EFFECT[r[0]] for r in rows)


def repeated(reps=200, **kw):
    """같은 설정으로 표본을 reps번 새로 뽑아 Wald 추정값의 흩어짐을 본다."""
    est = []
    for k in range(reps):
        rows = simulate(seed=1000 + k, **kw)
        est.append(wald(rows))
    est.sort()
    return {
        "median": median(est),
        "p10": est[int(0.10 * reps)],
        "p90": est[int(0.90 * reps) - 1],
        "sd": pvariance(est) ** 0.5,
    }


def report():
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
    v_strong = simulate(n=20000, direct=1.0)
    v_weak = simulate(n=20000, p_complier=0.03, direct=1.0)
    out["violate_strong"] = wald(v_strong)
    out["violate_weak"] = wald(v_weak)
    return out


if __name__ == "__main__":
    for k, v in report().items():
        print(k, v if not isinstance(v, float) else round(v, 3))
