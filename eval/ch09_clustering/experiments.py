"""모델 평가 9장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch09_clustering/experiments.py` 로 실행한다.

E0 첫 화면: 학생 6명. 번호만 엇갈린 완벽한 결과와 모두 한 명씩 따로 둔 결과
E1 (a) 번호 문제: 가우스 덩어리 3개 x 100점. 번호를 (0,1,2)->(2,0,1)로 바꾼 완벽한 결과, k-평균 결과
E2 (b) 우연 보정: 정답 5모둠 x 20명(n=100)에 무작위 라벨 k = 2, 5, 10, 20, 50, 시드 100개. n=1000도 함께
E3 (c) 라벨 없는 평가의 한계: 반달·동심원에서 정답 분할과 k-평균(k=2) 분할의 실루엣
E4 실루엣으로 k 고르기: k = 2..8, 덩어리(정답 k=4)·반달(정답 k=2)·동심원(정답 k=2), 시드 20개
E5 ARI와 AMI가 엇갈리나: 큰 모둠 2 + 작은 모둠 4, 합치기 후보 A와 쪼개기 후보 B, 시드 100개
E6 대조: 손 계산 fixture, Vinh의 E{I} = 0.4618, 헝가리안 = 전수 순열, 순열 모형 전수 평균, NumPy-PyTorch
결과는 results/ch09.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
"""

import json
import os
import platform
import sys
import time
from fractions import Fraction
from itertools import permutations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch09_cases as cases  # noqa: E402
import eval_ch09_metrics as m  # noqa: E402
import eval_ch09_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch09.json"
SEEDS = 100


def q(a, p=(5, 50, 95)):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), **{f"p{k}": float(np.percentile(a, k)) for k in p}}


def all_scores(y, c):
    return {"raw_acc": m.raw_accuracy(y, c), "matched_acc": m.matched_accuracy(y, c),
            "greedy_acc": m.greedy_majority_accuracy(y, c), "ri": m.rand_index(y, c), "ari": m.ari(y, c),
            "nmi_max": m.nmi(y, c, "max"), "nmi_sum": m.nmi(y, c, "arithmetic"),
            "ami_max": m.ami(y, c, "max"), "ami_sum": m.ami(y, c, "arithmetic")}


def e0_first_screen():
    t, p, s = cases.first_screen()
    n11, n10, n01, n00 = m.pair_counts(t, s)
    return {"truth": t.tolist(), "perfect": p.tolist(), "singletons": s.tolist(),
            "perfect_scores": all_scores(t, p), "singleton_scores": all_scores(t, s),
            "singleton_pairs": {"total": n11 + n10 + n01 + n00, "same_truth": n11 + n10, "apart_truth": n01 + n00,
                                "agree": n11 + n00},
            "singleton_expected_ri": m.expected_rand_index(t, s)}


def e1_permuted_blobs():
    X, y, c = cases.case_permuted_blobs(seed=0)
    mapping, hits = m.match_labels_hungarian(y, c)
    km = m.kmeans(X, 3, seed=0)
    km_map, km_hits = m.match_labels_hungarian(y, km)
    return {"n": len(y), "perfect_relabeled": all_scores(y, c), "mapping_cluster_to_true": {str(k): v for k, v in mapping.items()},
            "hits": hits, "kmeans": all_scores(y, km), "kmeans_mapping": {str(k): v for k, v in km_map.items()},
            "kmeans_hits": km_hits, "contingency_kmeans": m.contingency(y, km).tolist()}


def e2_random_labels():
    out = {}
    for n in (100, 1000):
        for k in (2, 5, 10, 20, 50):
            acc = {key: [] for key in ("nmi_max", "nmi_sum", "ami_max", "ami_sum", "ari", "ri", "matched_acc")}
            for s in range(SEEDS):
                t, p = cases.case_random_labels(n=n, k=k, seed=10_000 * (n == 1000) + 100 * k + s)
                acc["nmi_max"].append(m.nmi(t, p, "max")); acc["nmi_sum"].append(m.nmi(t, p, "arithmetic"))
                acc["ami_max"].append(m.ami(t, p, "max")); acc["ami_sum"].append(m.ami(t, p, "arithmetic"))
                acc["ari"].append(m.ari(t, p)); acc["ri"].append(m.rand_index(t, p))
                acc["matched_acc"].append(m.matched_accuracy(t, p))
            out[f"n{n}_k{k}"] = {key: q(v) for key, v in acc.items()}
    return {"n_true": 5, "seeds": SEEDS, "table": out}


def _wrong_vs_true(X, y, seed):
    km = m.kmeans(X, 2, seed=seed)
    D = m.pairwise_dist(X)
    return {"sil_true": m.silhouette(X, y, D), "sil_kmeans": m.silhouette(X, km, D), "ari_kmeans": m.ari(y, km),
            "matched_acc_kmeans": m.matched_accuracy(y, km)}


def e3_unlabeled_limit():
    out = {}
    for name, gen in (("moons", cases.case_moons), ("circles", cases.case_circles)):
        X, y = gen(seed=0)
        one = _wrong_vs_true(X, y, seed=0)
        cnt, gaps = 0, []
        for s in range(SEEDS):
            X, y = gen(seed=1000 + s)
            r = _wrong_vs_true(X, y, seed=s)
            cnt += r["sil_kmeans"] > r["sil_true"]
            gaps.append(r["sil_kmeans"] - r["sil_true"])
        out[name] = {"seed0": one, "seeds": SEEDS, "kmeans_beats_truth_by_silhouette": int(cnt), "gap": q(gaps)}
    return out


def e4_choose_k(n_seeds=20, ks=range(2, 9)):
    out = {}
    for name, gen, true_k in (("blobs4", lambda s: cases.case_blobs(k=4, seed=s), 4),
                              ("moons", lambda s: cases.case_moons(seed=s), 2),
                              ("circles", lambda s: cases.case_circles(seed=s), 2)):
        picks, hits, ari_at_pick, seed0 = [], 0, [], None
        for s in range(n_seeds):
            X, y = gen(2000 + s)
            D = m.pairwise_dist(X)
            sils, labs = {}, {}
            for k in ks:
                labs[k] = m.kmeans(X, k, seed=s)
                sils[k] = m.silhouette(X, labs[k], D)
            kbest = max(sils, key=sils.get)
            picks.append(int(kbest)); hits += kbest == true_k
            ari_at_pick.append(m.ari(y, labs[kbest]))
            if s == 0:
                seed0 = {str(k): v for k, v in sils.items()}
        out[name] = {"true_k": true_k, "seeds": n_seeds, "picked_true_k": int(hits),
                     "pick_counts": {str(k): picks.count(k) for k in ks}, "ari_at_pick": q(ari_at_pick),
                     "seed0_silhouette_by_k": seed0}
    return out


def e5_ari_vs_ami():
    t, a, b = cases.case_merge_vs_split(seed=0)
    seed0 = {"A_merge_small": all_scores(t, a), "B_split_big": all_scores(t, b)}
    c = {"ari_prefers_A": 0, "ami_max_prefers_A": 0, "ami_sum_prefers_A": 0, "nmi_sum_prefers_A": 0,
         "matched_prefers_A": 0, "ari_vs_ami_max_disagree": 0, "ari_vs_ami_sum_disagree": 0}
    for s in range(SEEDS):
        t, a, b = cases.case_merge_vs_split(seed=100 + s)
        sa, sb = all_scores(t, a), all_scores(t, b)
        pa = {k: sa[k] > sb[k] for k in sa}
        c["ari_prefers_A"] += pa["ari"]; c["ami_max_prefers_A"] += pa["ami_max"]; c["ami_sum_prefers_A"] += pa["ami_sum"]
        c["nmi_sum_prefers_A"] += pa["nmi_sum"]; c["matched_prefers_A"] += pa["matched_acc"]
        c["ari_vs_ami_max_disagree"] += pa["ari"] != pa["ami_max"]
        c["ari_vs_ami_sum_disagree"] += pa["ari"] != pa["ami_sum"]
    return {"sizes": [100, 100, 20, 20, 20, 20], "flip": 0.1, "seed0": seed0, "seeds": SEEDS,
            "counts": {k: int(v) for k, v in c.items()}}


def _perm_average(u, v, fn):
    """순열 모형의 기댓값을 전수로: v의 군집 크기는 그대로 두고 점 배치만 모든 순열로 바꾼다."""
    vals = [fn(u, np.asarray(v)[list(p)]) for p in permutations(range(len(v)))]
    return float(np.mean(vals))


def e6_checks():
    u, v = cases.fixture_ari_half()
    nu, nv = cases.fixture_nmi()
    X = cases.fixture_silhouette()
    a, b = cases.vinh_emi_sizes()
    rng = np.random.default_rng(9)
    hung = []
    for _ in range(200):
        w = rng.integers(0, 30, (rng.integers(2, 8), rng.integers(2, 8)))
        r, cc = m.hungarian_max(w)
        hung.append(abs(float(w[r, cc].sum()) - m.brute_force_max(w)))
    pu, pv = np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2, 2])
    t0 = time.perf_counter()
    perm = {"emi_bruteforce": _perm_average(pu, pv, m.mutual_info), "emi_formula": m.expected_mutual_info(pu, pv),
            "ari_mean_bruteforce": _perm_average(pu, pv, m.ari),
            "ri_mean_bruteforce": _perm_average(pu, pv, m.rand_index), "ri_expected_formula": m.expected_rand_index(pu, pv),
            "ami_mean_bruteforce": _perm_average(pu, pv, m.ami), "perms": 720, "seconds": time.perf_counter() - t0}
    diffs = {"ari": [], "mi": [], "emi": [], "ami_max": [], "ami_sum": [], "silhouette": [], "silhouette_cdist_default": []}
    for s in range(20):
        r2 = np.random.default_rng(100 + s)
        n = int(r2.integers(30, 200))
        x, y = r2.integers(0, r2.integers(2, 9), n), r2.integers(0, r2.integers(2, 9), n)
        P = r2.standard_normal((n, 3))
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            continue
        diffs["ari"].append(abs(m.ari(x, y) - tm.ari(x, y)))
        diffs["mi"].append(abs(m.mutual_info(x, y) - tm.mutual_info(x, y)))
        diffs["emi"].append(abs(m.expected_mutual_info(x, y) - tm.expected_mutual_info(x, y)))
        diffs["ami_max"].append(abs(m.ami(x, y, "max") - tm.ami(x, y, "max")))
        diffs["ami_sum"].append(abs(m.ami(x, y, "arithmetic") - tm.ami(x, y, "arithmetic")))
        diffs["silhouette"].append(abs(m.silhouette(P, x) - tm.silhouette(P, x)))
        diffs["silhouette_cdist_default"].append(abs(m.silhouette(P, x) - tm.silhouette(P, x, direct=False)))
    return {
        "ari_card": {"numpy": m.ari(u, v), "torch": tm.ari(u, v),
                     "exact": str(Fraction(m.ari(u, v)).limit_denominator(100))},
        "nmi_card": {"nmi_max": m.nmi(nu, nv, "max"), "nmi_sum": m.nmi(nu, nv, "arithmetic"), "mi_bits": m.mutual_info(nu, nv) / np.log(2)},
        "silhouette_card": {"split_good": m.silhouette_samples(X, [0, 0, 1, 1]).tolist(),
                            "split_bad": m.silhouette_samples(X, [0, 1, 1, 1]).tolist()},
        "vinh_emi": {"paper": 0.4618, "numpy": m.expected_mutual_info_sizes(a, b),
                     "torch": tm.expected_mutual_info(np.repeat(np.arange(10), a), np.repeat(np.arange(10), b)),
                     "n1000_numpy": m.expected_mutual_info_sizes([100] * 10, [20, 40, 60, 80, 100, 100, 120, 140, 160, 180])},
        "hungarian_vs_bruteforce_max_abs_diff": max(hung), "hungarian_cases": len(hung),
        "permutation_model": perm,
        "numpy_torch_max_abs_diff": {k: float(max(v)) for k, v in diffs.items()}, "numpy_torch_cases": len(diffs["ari"]),
    }


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                "machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "E0": e0_first_screen(), "E1": e1_permuted_blobs(), "E2": e2_random_labels(), "E3": e3_unlabeled_limit(),
        "E4": e4_choose_k(), "E5": e5_ari_vs_ami(), "E6": e6_checks(),
    }
    res["env"]["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
