"""1장 실습: 논문이 적은 요약 통계로 효과 크기와 매개 산술을 직접 다시 계산한다.

입력은 Wang, Si & Ali (2025)의 실험 1·2에 적힌 값이다(MDPI 공개본, 2026-11-18 확인).
계산 결과는 논문에 없는 직접 계산이다. 시간 지각을 흉내 내지 않는다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math

# 논문이 적은 값. 집단 크기가 같다고 가정해 합동 표준편차는 두 SD의 제곱 평균으로 쓴다.
EXP1 = {"dynamic": (3.12, 1.02), "static": (4.02, 1.09), "F": 24.38, "df1": 1, "df2": 196}
EXP2 = {"dynamic": (3.52, 1.14), "static": (4.67, 1.23), "F": 35.88, "df1": 1, "df2": 409}
# 집단별 인원은 논문 요약에 없어 가정한다. 실험 1은 198명을 99/99로, 실험 2는 합 411명을 205(움직이는 쪽)/206(멈춘 쪽)로 둔다.
EXP1_N = (99, 99)
EXP2_N = (205, 206)
EXP2_MEDIATION = {"a": -1.64, "b": 0.36, "ab_reported": -0.59, "c_prime": -0.09}


def d_from_summary(m1: float, s1: float, m2: float, s2: float) -> float:
    """두 평균의 차이를 두 SD의 제곱 평균의 제곱근으로 나눈다(집단 크기가 같다고 근사)."""
    sp = math.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / sp


def eta_squared_from_F(F: float, df1: int, df2: int) -> float:
    return F * df1 / (F * df1 + df2)


def d_from_F(F: float, df2: int) -> float:
    """보고된 F(자유도 1과 df2)에서 거꾸로 구한 d의 크기. 두 집단 크기가 같다고 보고 d=2t/√df2, t=√F."""
    return 2 * math.sqrt(F / df2)


def f_from_summary(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> dict:
    """두 집단 평균, SD, 인원으로 다시 구한 일원 분산분석 F, 부분 η², d(인원 가중 합동 SD)."""
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    t = (m1 - m2) / math.sqrt(sp2 * (1 / n1 + 1 / n2))
    f = t * t
    return {"F": f, "eta2": f / (f + n1 + n2 - 2), "d": (m1 - m2) / math.sqrt(sp2)}


def mediation_check(a: float, b: float, c_prime: float, mean_diff: float) -> dict:
    ab = a * b
    return {"ab": ab, "total": c_prime + ab, "mean_diff": mean_diff, "gap": mean_diff - (c_prime + ab)}


def main() -> None:
    for name, e in (("실험 1", EXP1), ("실험 2", EXP2)):
        d = d_from_summary(*e["dynamic"], *e["static"])
        sp = math.sqrt((e["dynamic"][1] ** 2 + e["static"][1] ** 2) / 2)
        print(f"{name}: 합친 SD {sp:.3f}, 평균 차이 {e['dynamic'][0] - e['static'][0]:.2f}, 평균·SD에서 구한 d={d:.2f}")
    for name, e, rep_eta in (("실험 1", EXP1, 0.12), ("실험 2", EXP2, 0.08)):
        eta = eta_squared_from_F(e["F"], e["df1"], e["df2"])
        print(f"{name} 보고 F={e['F']}에서: η²={eta:.3f}(논문 보고 {rep_eta}), 거꾸로 구한 d={d_from_F(e['F'], e['df2']):.2f}")
    for name, e, n in (("실험 1", EXP1, EXP1_N), ("실험 2", EXP2, EXP2_N)):
        r = f_from_summary(*e["dynamic"], n[0], *e["static"], n[1])
        print(f"{name} 평균·SD와 집단 크기 {n[0]}/{n[1]}로 다시 만든 값: F={r['F']:.1f}(보고 {e['F']}), η²={r['eta2']:.3f}, d={r['d']:.2f}")
    m = EXP2_MEDIATION
    r = mediation_check(m["a"], m["b"], m["c_prime"], EXP2["dynamic"][0] - EXP2["static"][0])
    print(f"실험 2 매개: a×b={r['ab']:.4f}(논문 {m['ab_reported']}), c'+ab={r['total']:.4f}, 평균 차이 {r['mean_diff']:.2f}, 어긋남 {r['gap']:.2f}")


if __name__ == "__main__":
    main()
