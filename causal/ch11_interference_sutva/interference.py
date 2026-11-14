"""11장 간섭과 SUTVA. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 친구 초대 쿠폰.

고객 i의 지출 Y_i = 기본 + tau*T_i + gamma*(이웃 중 처치받은 비율) + 잡음.
tau는 내 쿠폰의 직접 효과, gamma는 이웃이 받은 쿠폰이 나에게 번지는 정도(음수면 경쟁/잠식).
진짜 '전체 출시' 효과 = 모두가 쿠폰을 받았을 때 - 아무도 안 받았을 때 = tau + gamma (이웃 비율이 1과 0이므로).
"""
import random
from statistics import mean

K, M = 20, 50  # 커뮤니티 20개, 커뮤니티마다 50명 = 1000명(가상)


def make_graph(seed=11, p_in=0.20, p_out=0.002):
    rng = random.Random(seed)
    n = K * M
    comm = [i // M for i in range(n)]
    nbr = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p:
                nbr[i].append(j)
                nbr[j].append(i)
    return comm, nbr


def outcome(i, t, nbr, base, noise, tau, gamma):
    ns = nbr[i]
    frac = (sum(t[j] for j in ns) / len(ns)) if ns else 0.0
    return base[i] + tau * t[i] + gamma * frac + noise[i]


def draw_units(n, seed):
    rng = random.Random(seed)
    base = [50 + rng.gauss(0, 5) for _ in range(n)]
    noise = [rng.gauss(0, 3) for _ in range(n)]
    return base, noise


def true_global_effect(nbr, base, noise, tau, gamma):
    """같은 사람·같은 잡음에서 전원 처치 vs 전원 미처치. 이웃이 없는 사람은 gamma가 0으로 들어간다."""
    n = len(nbr)
    all_t = [1] * n
    all_c = [0] * n
    return mean(outcome(i, all_t, nbr, base, noise, tau, gamma) - outcome(i, all_c, nbr, base, noise, tau, gamma)
                for i in range(n))


def diff_means(y, t, idx):
    a = [y[i] for i in idx if t[i] == 1]
    b = [y[i] for i in idx if t[i] == 0]
    return mean(a) - mean(b)


def individual_design(comm, nbr, base, noise, tau, gamma, seed):
    rng = random.Random(seed)
    n = len(nbr)
    t = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
    y = [outcome(i, t, nbr, base, noise, tau, gamma) for i in range(n)]
    return diff_means(y, t, range(n))


def cluster_design(comm, nbr, base, noise, tau, gamma, seed):
    rng = random.Random(seed)
    n = len(nbr)
    ids = list(range(K))
    rng.shuffle(ids)
    treated_comm = set(ids[: K // 2])
    t = [1 if comm[i] in treated_comm else 0 for i in range(n)]
    y = [outcome(i, t, nbr, base, noise, tau, gamma) for i in range(n)]
    return diff_means(y, t, range(n))


def repeated(tau, gamma, reps=100, seed=11):
    comm, nbr = make_graph(seed)
    base, noise = draw_units(len(nbr), seed + 1)
    truth = true_global_effect(nbr, base, noise, tau, gamma)
    ind = [individual_design(comm, nbr, base, noise, tau, gamma, 5000 + r) for r in range(reps)]
    clu = [cluster_design(comm, nbr, base, noise, tau, gamma, 9000 + r) for r in range(reps)]
    return {"truth": truth, "individual": mean(ind), "cluster": mean(clu),
            "ind_sd": (mean([(x - mean(ind)) ** 2 for x in ind])) ** 0.5,
            "clu_sd": (mean([(x - mean(clu)) ** 2 for x in clu])) ** 0.5}


def neighbor_share(comm, nbr, reps=100):
    """처치받은 고객과 못 받은 고객이 '이웃 중 처치받은 비율'을 평균 얼마나 갖는지(배정 reps번 평균)."""
    n = len(nbr)

    def frac(i, t):
        return sum(t[j] for j in nbr[i]) / len(nbr[i])

    out = {}
    for name in ("individual", "cluster"):
        a, b = [], []
        for r in range(reps):
            if name == "individual":
                rng = random.Random(5000 + r)
                t = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
            else:
                rng = random.Random(9000 + r)
                ids = list(range(K))
                rng.shuffle(ids)
                tc = set(ids[: K // 2])
                t = [1 if comm[i] in tc else 0 for i in range(n)]
            a.append(mean(frac(i, t) for i in range(n) if t[i] == 1))
            b.append(mean(frac(i, t) for i in range(n) if t[i] == 0))
        out[name + "_treated"] = mean(a)
        out[name + "_control"] = mean(b)
    return out


def report():
    comm, nbr = make_graph()
    deg = [len(x) for x in nbr]
    cross = sum(1 for i in range(len(nbr)) for j in nbr[i] if comm[i] != comm[j]) / sum(deg)
    return {
        "mean_degree": mean(deg),
        "isolated": sum(1 for d in deg if d == 0),
        "cross_edge_share": cross,
        "neighbor_share": neighbor_share(comm, nbr),
        "no_spillover": repeated(5.0, 0.0),
        "spillover_pos": repeated(5.0, 4.0),
        "spillover_neg": repeated(5.0, -4.0),
    }


if __name__ == "__main__":
    r = report()
    for k, v in r.items():
        if isinstance(v, dict):
            print(k, {a: round(b, 2) for a, b in v.items()})
        else:
            print(k, round(v, 3) if isinstance(v, float) else v)
