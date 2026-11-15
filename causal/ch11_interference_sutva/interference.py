"""11장 간섭과 SUTVA. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 친구 초대 쿠폰.

고객 i의 지출 Y_i = 기본 + tau*T_i + gamma*(이웃 중 처치받은 비율) + 잡음.
tau는 내 쿠폰의 직접 효과, gamma는 이웃이 받은 쿠폰이 나에게 번지는 정도(음수면 경쟁/잠식).
진짜 '전체 출시' 효과 = 모두가 쿠폰을 받았을 때 - 아무도 안 받았을 때 = tau + gamma (이웃 비율이 1과 0이므로).

무엇을 재는가: SUTVA(Stable Unit Treatment Value Assumption, 내 결과가 다른 사람의 처치에 영향받지 않는다는 가정)가 깨질 때,
개인 단위 무작위 A/B 추정과 커뮤니티 단위(클러스터) 무작위 추정이 "전체 출시 효과"를 얼마나 맞히는지 비교한다.
입력: make_graph로 만든 커뮤니티 20개 x 50명의 친구 그래프(커뮤니티 안 연결 확률 0.20, 밖 0.002), tau(직접 효과), gamma(이웃 효과).
출력: report()의 dict(그래프 모양, 설계별 평균 추정과 표준편차). 테스트의 기대값(평균 이웃 수 약 11.7, 진짜 효과 5/9/1 등)이다(장 본문 숫자와의 대응은 README로 확인 필요).
단위는 가상의 지출 단위.
"""
import random
from statistics import mean

K, M = 20, 50  # 커뮤니티 20개, 커뮤니티마다 50명 = 1000명(가상)


def make_graph(seed=11, p_in=0.20, p_out=0.002):
    """커뮤니티 안은 촘촘(0.20), 커뮤니티 사이는 드문(0.002) 무방향 그래프. (커뮤니티 번호 목록, 이웃 목록)을 돌려준다."""
    rng = random.Random(seed)
    n = K * M
    comm = [i // M for i in range(n)]  # 0..49는 커뮤니티 0, 50..99는 커뮤니티 1 ...
    nbr = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):  # 쌍(i, j)을 한 번만 보고 양쪽 이웃 목록에 모두 넣는다
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p:
                nbr[i].append(j)
                nbr[j].append(i)
    return comm, nbr


def outcome(i, t, nbr, base, noise, tau, gamma):
    """사람 i의 지출. 내 처치 t[i]의 직접 효과 tau + 이웃 중 처치 비율 x gamma(간섭)."""
    ns = nbr[i]
    frac = (sum(t[j] for j in ns) / len(ns)) if ns else 0.0  # 이웃이 없으면 0으로 나누지 않고 0(이 그래프에선 고립 노드 0명)
    return base[i] + tau * t[i] + gamma * frac + noise[i]


def draw_units(n, seed):
    """사람별 기본 지출(평균 50, SD 5)과 잡음(SD 3)을 미리 뽑아 둔다. 설계를 바꿔도 같은 사람·같은 잡음을 쓰기 위함."""
    rng = random.Random(seed)
    base = [50 + rng.gauss(0, 5) for _ in range(n)]
    noise = [rng.gauss(0, 3) for _ in range(n)]
    return base, noise


def true_global_effect(nbr, base, noise, tau, gamma):
    """같은 사람·같은 잡음에서 전원 처치 vs 전원 미처치. 이웃이 없는 사람은 gamma가 0으로 들어간다."""
    # 같은 base, noise에서 처치만 바꿔 비교하므로 차이는 tau + gamma x (이웃 비율 1 - 0) 그대로다
    n = len(nbr)
    all_t = [1] * n
    all_c = [0] * n
    return mean(outcome(i, all_t, nbr, base, noise, tau, gamma) - outcome(i, all_c, nbr, base, noise, tau, gamma)
                for i in range(n))


def diff_means(y, t, idx):
    """idx 안에서 처치군 평균 - 대조군 평균."""
    a = [y[i] for i in idx if t[i] == 1]
    b = [y[i] for i in idx if t[i] == 0]
    return mean(a) - mean(b)


def individual_design(comm, nbr, base, noise, tau, gamma, seed):
    """개인 단위 동전 배정 A/B. 이웃의 약 절반이 처치라 gamma 몫이 처치군과 대조군에 똑같이 들어가 상쇄된다(= tau만 잰다)."""
    rng = random.Random(seed)
    n = len(nbr)
    t = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
    y = [outcome(i, t, nbr, base, noise, tau, gamma) for i in range(n)]
    return diff_means(y, t, range(n))


def cluster_design(comm, nbr, base, noise, tau, gamma, seed):
    """커뮤니티 단위 무작위 배정. 처치 커뮤니티 사람은 이웃이 대부분 처치라 gamma 몫을 크게 얻는다. 커뮤니티 사이 간선이 있어 완전하지는 않다."""
    rng = random.Random(seed)
    n = len(nbr)
    ids = list(range(K))
    rng.shuffle(ids)
    treated_comm = set(ids[: K // 2])
    t = [1 if comm[i] in treated_comm else 0 for i in range(n)]
    y = [outcome(i, t, nbr, base, noise, tau, gamma) for i in range(n)]
    return diff_means(y, t, range(n))


def repeated(tau, gamma, reps=100, seed=11):
    """같은 그래프·사람에서 배정만 reps번 바꿔 두 설계의 평균 추정과 표준편차(모표준편차, 분모 n)를 모은다."""
    comm, nbr = make_graph(seed)
    base, noise = draw_units(len(nbr), seed + 1)
    truth = true_global_effect(nbr, base, noise, tau, gamma)
    # 시드를 5000대(개인), 9000대(클러스터)로 구분해 두 설계의 배정이 서로 겹치지 않게 한다
    ind = [individual_design(comm, nbr, base, noise, tau, gamma, 5000 + r) for r in range(reps)]
    clu = [cluster_design(comm, nbr, base, noise, tau, gamma, 9000 + r) for r in range(reps)]
    return {"truth": truth, "individual": mean(ind), "cluster": mean(clu),
            "ind_sd": (mean([(x - mean(ind)) ** 2 for x in ind])) ** 0.5,
            "clu_sd": (mean([(x - mean(clu)) ** 2 for x in clu])) ** 0.5}


def neighbor_share(comm, nbr, reps=100):
    """처치받은 고객과 못 받은 고객이 '이웃 중 처치받은 비율'을 평균 얼마나 갖는지(배정 reps번 평균)."""
    # 처치군/대조군이 평균 얼마나 많은 처치 이웃을 갖는지. 개인 배정은 둘이 같고, 클러스터 배정은 크게 벌어진다
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
    """그래프 모양과 세 가지 간섭 시나리오(gamma = 0, +4, -4; tau = 5)의 설계별 추정을 모은다."""
    comm, nbr = make_graph()
    deg = [len(x) for x in nbr]
    # 전체 간선 끝점 중 다른 커뮤니티로 가는 비율. 클러스터 설계가 덜 맞는 이유(경계 오염)의 크기
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
