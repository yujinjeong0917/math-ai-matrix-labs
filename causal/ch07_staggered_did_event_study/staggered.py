"""7장 실습: 시차 도입 이중차분, 이벤트스터디, 이원고정효과(TWFE)의 가중치 함정.
표준 라이브러리만 사용합니다. 진짜 효과는 데이터를 만들 때 정해 두었어요.
"""
import math
import random

# ---------- 손계산: 지역 2곳, 3주, 대조 지역 없음 ----------
# A는 2주차부터, B는 3주차부터 쿠폰. 쿠폰이 없을 때는 두 지역 모두 매주 평평(100).
# 진짜 효과는 쿠폰 시작 후 첫 주 +10, 둘째 주 +20 (A), 첫 주 +10 (B).
HAND_A = [100.0, 110.0, 120.0]
HAND_B = [100.0, 100.0, 110.0]


def hand_two_by_two(late, early, t0, t1):
    """late(처치 중인 쪽으로 보는 지역)의 t0->t1 변화에서 early의 같은 기간 변화를 뺀다."""
    return (late[t1] - late[t0]) - (early[t1] - early[t0])


# ---------- 난수 ----------
def normal(rng):
    u1, u2 = rng.random(), rng.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


# ---------- 패널 만들기 ----------
GROUPS = {"A": 3, "B": 7, "C": None}  # 쿠폰 시작 주(지역 A 3주차, B 7주차, C는 끝까지 없음)
REGION_LEVEL = {"A": 20.0, "B": 0.0, "C": 10.0}
T = 10
STORES = 20


def effect(e):
    """쿠폰 시작 후 e주째의 진짜 효과. 시간이 갈수록 커진다(가상, 천 원)."""
    return 3.0 * (e + 1) if e >= 0 else 0.0


def make_panel(seed=11, groups=GROUPS, noise=2.0):
    rng = random.Random(seed)
    rows = []
    sid = 0
    for g, start in groups.items():
        for _ in range(STORES):
            fe = REGION_LEVEL[g] + 5.0 * normal(rng)
            for t in range(1, T + 1):
                d = 1 if (start is not None and t >= start) else 0
                tau = effect(t - start) if d else 0.0
                y = 100.0 + fe + 1.5 * t + noise * normal(rng) + tau
                rows.append({"id": sid, "g": g, "t": t, "d": d, "y": y, "tau": tau})
            sid += 1
    return rows


def true_average_effect(rows):
    tr = [r["tau"] for r in rows if r["d"] == 1]
    return sum(tr) / len(tr)


# ---------- TWFE 계수와 처치 칸별 가중치 ----------
def twfe(rows):
    """균형 패널에서 단위·시간 고정효과를 뺀 처치 변수 D~로 구한 계수와 처치 칸 가중치."""
    ids = sorted({r["id"] for r in rows})
    ts = sorted({r["t"] for r in rows})
    n, tt = len(ids), len(ts)
    dsum_i = {i: 0.0 for i in ids}
    dsum_t = {t: 0.0 for t in ts}
    total = 0.0
    for r in rows:
        dsum_i[r["id"]] += r["d"]
        dsum_t[r["t"]] += r["d"]
        total += r["d"]
    dbar = total / (n * tt)
    num = den = 0.0
    wt = []
    for r in rows:
        dd = r["d"] - dsum_i[r["id"]] / tt - dsum_t[r["t"]] / n + dbar
        num += dd * r["y"]
        den += dd * dd
        if r["d"] == 1:
            wt.append((r, dd))
    treated_sum = sum(v for _, v in wt)
    weights = [(r, v / treated_sum) for r, v in wt]
    return num / den, weights


def weight_summary(weights):
    """그룹(지역)별 가중치 합과 음수 가중치 합."""
    by = {}
    for r, w in weights:
        by.setdefault(r["g"], 0.0)
        by[r["g"]] += w
    neg = sum(w for _, w in weights if w < 0)
    n_neg = sum(1 for _, w in weights if w < 0)
    return by, neg, n_neg, len(weights)


def weight_parts(weights):
    """처치 칸을 (A 3~6주차, A 7~10주차, B)로 나눠 (이름, 가중치 합, 그 칸들의 진짜 평균 효과)."""
    parts = {"A 3~6주차": [], "A 7~10주차": [], "B 7~10주차": []}
    for r, w in weights:
        key = "B 7~10주차" if r["g"] == "B" else ("A 7~10주차" if r["t"] >= GROUPS["B"] else "A 3~6주차")
        parts[key].append((w, r["tau"]))
    return [(k, sum(w for w, _ in v), sum(t for _, t in v) / len(v)) for k, v in parts.items()]


def weighted_average_effect(weights):
    """TWFE가 사실상 평균 내는 진짜 효과 = Σ weight * tau (노이즈 없는 몫)."""
    return sum(w * r["tau"] for r, w in weights)


# ---------- 깨끗한 비교: 그룹-시점 효과(Callaway-Sant'Anna 방식의 단순판) ----------
def cell_means(rows):
    s, c = {}, {}
    for r in rows:
        k = (r["g"], r["t"])
        s[k] = s.get(k, 0.0) + r["y"]
        c[k] = c.get(k, 0) + 1
    return {k: s[k] / c[k] for k in s}


def att_gt(rows, groups=GROUPS, control="C"):
    """ATT(g, t): 도입 직전 주(start-1)를 기준으로 처치 지역 변화 - 아직 쿠폰 없는 대조 지역(C) 변화.
    t가 도입 전이면 '가짜 효과'(사전추세 검사)가 된다."""
    m = cell_means(rows)
    out = {}
    for g, start in groups.items():
        if start is None:
            continue
        base = start - 1
        for t in range(1, T + 1):
            if t == base:
                continue
            out[(g, t)] = (m[(g, t)] - m[(g, base)]) - (m[(control, t)] - m[(control, base)])
    return out


def event_study(att, groups=GROUPS):
    """상대 시점 e = t - start 별로 그룹 평균(그룹 크기가 같아 단순 평균)."""
    by = {}
    for (g, t), v in att.items():
        by.setdefault(t - groups[g], []).append(v)
    return {e: sum(v) / len(v) for e, v in sorted(by.items())}


def overall_att(att, groups=GROUPS):
    """처치 칸(그룹, 시점 >= 도입)의 ATT(g,t)를 칸 수대로 단순 평균."""
    v = [x for (g, t), x in att.items() if t >= groups[g]]
    return sum(v) / len(v)


def main():
    print("[손계산: 2개 지역, 3주, 대조 지역 없음]")
    print(f"진짜 효과: A는 2주차 +10, 3주차 +20, B는 3주차 +10")
    print(f"3주차에 B의 변화를 A의 변화와 비교(2->3주): {hand_two_by_two(HAND_B, HAND_A, 1, 2):+.0f}")
    print(f"2주차에 A의 변화를 B의 변화와 비교(1->2주): {hand_two_by_two(HAND_A, HAND_B, 0, 1):+.0f}")

    rows = make_panel()
    print(f"\n[패널] 지역 3곳 x 가게 {STORES}곳 x {T}주. A는 {GROUPS['A']}주차, B는 {GROUPS['B']}주차에 쿠폰, C는 없음")
    print(f"처치 칸 수 {sum(r['d'] for r in rows)}, 진짜 평균 효과(처치 칸 단순 평균) {true_average_effect(rows):.2f}")
    beta, wts = twfe(rows)
    by, neg, n_neg, n = weight_summary(wts)
    print(f"TWFE 계수 {beta:.2f}")
    print(f"TWFE 가중치 합: A {by['A']:.3f}, B {by['B']:.3f}, 음수 가중치 칸 {n_neg}/{n}, 음수 가중치 합 {neg:.3f}")
    print(f"가중치가 가리키는 진짜 효과 평균 {weighted_average_effect(wts):.2f}")
    for name, w_sum, tau_mean in weight_parts(wts):
        print(f"  {name}: 가중치 합 {w_sum:+.3f}, 진짜 평균 효과 {tau_mean:.2f}, 기여 {w_sum * tau_mean:+.4f}")
    quiet_beta, _ = twfe(make_panel(noise=0.0))
    print(f"잡음을 0으로 둔 같은 패널의 TWFE 계수 {quiet_beta:.2f} (잡음이 있으면 {beta:.2f})")
    a_late = [(r["t"], w) for r, w in wts if r["g"] == "A" and r["t"] >= GROUPS["B"]]
    print(f"B가 도입한 뒤(7주차부터) A의 칸 {len(a_late)}개: 가중치 합 {sum(w for _, w in a_late):.3f}, 모두 음수 {all(w < 0 for _, w in a_late)}")
    a_true = sum(r["tau"] for r, w in wts if r["g"] == "A" and r["t"] >= GROUPS["B"]) / len(a_late)
    print(f"그 칸들의 진짜 평균 효과 {a_true:.2f}")

    att = att_gt(rows)
    es = event_study(att)
    print("\n[깨끗한 비교] 아직 쿠폰 없는 C를 대조로, 도입 직전 주를 기준")
    print("상대 시점 e: 추정 / 진짜  (e<0은 가짜 효과, 0이어야 함)")
    for e, v in es.items():
        print(f"  e={e:+d}: {v:+.2f} / {effect(e):+.1f}")
    print(f"전체 ATT(처치 칸 단순 평균) {overall_att(att):.2f}")

    print("\n[대조 지역 C를 빼고 A, B만 쓴 TWFE]")
    two = [r for r in rows if r["g"] != "C"]
    beta2, w2 = twfe(two)
    by2, neg2, n_neg2, n2 = weight_summary(w2)
    print(f"진짜 평균 효과 {true_average_effect(two):.2f}, TWFE 계수 {beta2:.2f}, 음수 가중치 칸 {n_neg2}/{n2}")
    print(f"가중치가 가리키는 진짜 효과 평균 {weighted_average_effect(w2):.2f} (나머지 차이는 잡음)")
    for name, w_sum, tau_mean in weight_parts(w2):
        print(f"  {name}: 가중치 합 {w_sum:+.3f}, 진짜 평균 효과 {tau_mean:.2f}, 기여 {w_sum * tau_mean:+.2f}")

    print("\n[효과가 시간에 따라 안 변하고 +8로 일정할 때]")
    flat = make_panel()
    for r in flat:
        if r["d"] == 1:
            r["y"] += 8.0 - r["tau"]
            r["tau"] = 8.0
    bflat, _ = twfe(flat)
    print(f"TWFE 계수 {bflat:.2f} (진짜 8.00)")
    quiet = make_panel(noise=0.0)
    for r in quiet:
        if r["d"] == 1:
            r["y"] += 8.0 - r["tau"]
            r["tau"] = 8.0
    print(f"잡음을 0으로 두면 {twfe(quiet)[0]:.2f}")


if __name__ == "__main__":
    main()
