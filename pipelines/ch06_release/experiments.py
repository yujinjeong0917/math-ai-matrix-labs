"""6장 비교 실험. `uv run python pipelines/ch06_release/experiments.py` 로 실행한다(네트워크·Jenkins·쿠버네티스 불필요).

E1 오프라인과 운영의 차이: v2는 오프라인 검증에서 v1보다 정확하다. 운영에서는 부산 앱이 usage_drop을 퍼센트로 보내
   그 지역에서만 크게 틀린다. 지역별 정확도와 모델 카드(results/model_card_v2.md)를 남긴다.
E2 배포 방식 비교: 전체 교체(되돌리기 지연 500·2000요청), 블루그린, 카나리 p = 0.01·0.05·0.2.
   정답이 늦게 오는 정도(label_delay 0·1000·3000요청)별로 늘어난 오류(피해), 결정까지 요청 수, v2가 받은 요청 수. 시드 10개.
E3 감시 지표: 전체 오류율만 볼 때와 지역별로도 볼 때(구간 가드). 같은 나쁜 v2를 올려 버리는 횟수.
E4 오탐: v2가 v1과 똑같을 때 되돌리는 비율. 문턱 z, 구간 가드, 여러 번 보기와 한 번 보기, 시드 200개.
   같은 설정에서 나쁜 v2를 잡는 비율도 함께. 그리고 배포 날 세상이 바뀌면(shift) 전후 비교와 동시 대조군이 어떻게 다른지.
E5 되돌리기와 공유 데이터: 요금 열을 그 자리에서 바꾼 경우와 새 열을 더한 경우, 되돌린 v1의 정확도.
E6 NumPy와 PyTorch 대조: 같은 특성 행렬에서 가중치 차이, 예측 일치, 카나리 피해 일치.
E7 고정 배정: 해시 라우터와 요청마다 동전 던지기에서 한 사용자가 두 버전을 오가는 비율.

결과는 results/ch06.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch06_card as card  # noqa: E402
import pipelines_ch06_data as D  # noqa: E402
import pipelines_ch06_model_np as M  # noqa: E402
import pipelines_ch06_model_torch as MT  # noqa: E402
import pipelines_ch06_schema as SC  # noqa: E402
from pipelines_ch06_release_gate import DEFAULT_GUARD, OVERALL_ONLY  # noqa: E402
from pipelines_ch06_router import bucket, route, route_random  # noqa: E402
from pipelines_ch06_simulate import metrics_of, simulate_traffic  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch06.json"
N_REQ = 10_000
SEEDS = range(10)
STRATS = [
    ("full_R500", "full", 1.0, 500),
    ("full_R2000", "full", 1.0, 2000),
    ("bluegreen", "bluegreen", 1.0, 0),
    ("canary_0.01", "canary", 0.01, 0),
    ("canary_0.05", "canary", 0.05, 0),
    ("canary_0.2", "canary", 0.2, 0),
]
DELAYS = (0, 1000, 3000)


def r4(x):
    return round(float(x), 4)


def train_models(fit=M.fit_logistic):
    t = D.make_table(0, 4000)
    tr, va = D.split(t, 3000, 0)
    return tr, va, M.train(tr, D.NUMERIC_V1, fit=fit), M.train(tr, D.NUMERIC_V2, fit=fit)


def traffic(seed, v1, v2, shift=0.0, skew=True):
    tf = D.serve_view(D.make_table(1000 + seed, N_REQ, shift=shift), skew=skew)
    bt = D.serve_view(D.make_table(5000 + seed, 2000), skew=skew)  # 배포 전 v1 구간(전후 비교의 기준)
    return {
        "c1": M.correct(v1, tf), "c2": M.correct(v2, tf), "seg": tf["region"],
        "users": [f"s{seed}-u{i}" for i in range(N_REQ)],
        "base": metrics_of(M.correct(v1, bt), bt["region"]),
    }


def summarize(rows):
    harm = [r["harm"] for r in rows]
    dec = [r["decided_at"] for r in rows if r["decided_at"] is not None]
    return {
        "harm_mean": r4(np.mean(harm)), "harm_std": r4(np.std(harm)),
        "harm_busan_mean": r4(np.mean([r["harm_by_segment"].get(D.SKEW_REGION, 0) for r in rows])),
        "harm_min": int(min(harm)), "harm_max": int(max(harm)),
        "served_v2_mean": r4(np.mean([r["served_v2"] for r in rows])),
        "decided_at_mean": r4(np.mean(dec)) if dec else None,
        "decided_at_min": int(min(dec)) if dec else None,
        "decided_at_max": int(max(dec)) if dec else None,
        "labeled_v2_at_decision_mean": r4(np.mean([r["v2_labeled_at_decision"] for r in rows if r["decided_at"] is not None])) if dec else None,
        "decisions": {k: sum(r["decision"] == k for r in rows) for k in ("rollback", "promote", "none")},
    }


def e1(tr, va, v1, v2):
    off1, off2 = card.by_region(v1, va), card.by_region(v2, va)
    online = {"v1": {}, "v2": {}}
    deltas, deltas_busan = [], []
    for s in SEEDS:
        tf = D.serve_view(D.make_table(1000 + s, N_REQ))
        a, b = card.by_region(v1, tf), card.by_region(v2, tf)
        for k in a:
            online["v1"].setdefault(k, []).append(a[k]["acc"])
            online["v2"].setdefault(k, []).append(b[k]["acc"])
        c1, c2 = M.correct(v1, tf), M.correct(v2, tf)
        deltas.append(float(np.mean(c1 - c2)))
        m = tf["region"] == D.SKEW_REGION
        deltas_busan.append(float(np.mean((c1 - c2)[m])))
    tf0 = D.serve_view(D.make_table(1000, N_REQ))
    m0 = tf0["region"] == D.SKEW_REGION
    p_busan = M.predict(v2, tf0)[m0]
    (HERE / "results" / "model_card_v2.md").write_text(
        card.render("churn v2 (usage_drop 추가)", off1, off2, "오프라인 검증 1,000행, data_seed 0", D.NUMERIC_V2)
    )
    return {
        "train_rows": len(tr["churned"]), "valid_rows": len(va["churned"]),
        "offline": {"v1": {k: r4(v["acc"]) for k, v in off1.items()}, "v2": {k: r4(v["acc"]) for k, v in off2.items()},
                    "n": {k: v["n"] for k, v in off1.items()}},
        "online_mean": {ver: {k: r4(np.mean(v)) for k, v in d.items()} for ver, d in online.items()},
        "delta_e_mean": r4(np.mean(deltas)), "delta_e_min": r4(min(deltas)), "delta_e_max": r4(max(deltas)),
        "delta_e_busan_mean": r4(np.mean(deltas_busan)),
        "busan_share_seed0": r4(m0.mean()),
        "busan_v2_predicts_churn_seed0": r4(p_busan.mean()),
        "busan_churn_rate_seed0": r4(tf0["churned"][m0].mean()),
        "usage_drop_v2_weight": r4(v2["w"][D.NUMERIC_V2.index("usage_drop")]),
        "usage_drop_train_mean_std": [r4(v2["prep"]["mean"]["usage_drop"]), r4(v2["prep"]["std"]["usage_drop"])],
    }


def e2(trafs):
    out = {}
    for L in DELAYS:
        for name, kind, p, R in STRATS:
            rows = []
            for s, T in trafs.items():
                r = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], kind, DEFAULT_GUARD, p=p,
                                     baseline=T["base"], rollback_delay=R, label_delay=L)
                r["delta_e"] = float(np.mean(T["c1"] - T["c2"]))
                rows.append(r)
            sm = summarize(rows)
            sm["harm_formula_mean"] = r4(np.mean([r["served_v2"] * r["delta_e"] for r in rows]))
            out[f"L{L}/{name}"] = sm
    return out


def e3(trafs):
    out = {}
    for gname, g in (("overall_only", OVERALL_ONLY), ("with_segments", DEFAULT_GUARD)):
        for name, kind, p, R in STRATS:
            if name == "full_R2000":
                continue
            rows = [simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], kind, g, p=p, baseline=T["base"],
                                     rollback_delay=R) for T in trafs.values()]
            out[f"{gname}/{name}"] = summarize(rows)
    return out


def e4(v1, v2):
    n_seeds = 200
    configs = {
        "z3_overall_repeated": (dict(OVERALL_ONLY, z=3.0), None),
        "z3_overall_single": (dict(OVERALL_ONLY, z=3.0), 400),
        "z3_segments_repeated": (dict(DEFAULT_GUARD, z=3.0), None),
        "z3_segments_single": (dict(DEFAULT_GUARD, z=3.0), 400),
        "z2_overall_repeated": (dict(OVERALL_ONLY, z=2.0), None),
        "z2_overall_single": (dict(OVERALL_ONLY, z=2.0), 400),
        "z2_segments_repeated": (dict(DEFAULT_GUARD, z=2.0), None),
        "z2_segments_single": (dict(DEFAULT_GUARD, z=2.0), 400),
    }
    same = {k: 0 for k in configs}
    bad = {k: 0 for k in configs}
    for s in range(n_seeds):
        T = traffic(100 + s, v1, v2)
        for k, (g, single) in configs.items():
            r0 = simulate_traffic(T["c1"], T["c1"], T["users"], T["seg"], "canary", g, p=0.2, single_look_at=single)
            same[k] += r0["decision"] == "rollback"
            r1 = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "canary", g, p=0.2, single_look_at=single)
            bad[k] += r1["decision"] == "rollback"
    # 배포 날 세상이 바뀐 경우: v2 = v1(똑같은 모델), 운영 구간에만 shift=1.0. 시드 50개.
    shift_seeds = 50
    world = {"bluegreen_no_shift": 0, "bluegreen_shift": 0, "canary_0.2_shift": 0}
    v1_err = {"before": [], "after_shift": []}
    for s in range(shift_seeds):
        T0 = traffic(300 + s, v1, v2)
        T1 = traffic(300 + s, v1, v2, shift=1.0)
        world["bluegreen_no_shift"] += simulate_traffic(T0["c1"], T0["c1"], T0["users"], T0["seg"], "bluegreen",
                                                        DEFAULT_GUARD, baseline=T0["base"])["decision"] == "rollback"
        world["bluegreen_shift"] += simulate_traffic(T1["c1"], T1["c1"], T1["users"], T1["seg"], "bluegreen",
                                                     DEFAULT_GUARD, baseline=T1["base"])["decision"] == "rollback"
        world["canary_0.2_shift"] += simulate_traffic(T1["c1"], T1["c1"], T1["users"], T1["seg"], "canary",
                                                      DEFAULT_GUARD, p=0.2)["decision"] == "rollback"
        v1_err["before"].append(T1["base"]["errors"] / T1["base"]["n"])
        v1_err["after_shift"].append(float(1 - T1["c1"].mean()))
    return {
        "n_seeds": n_seeds, "p": 0.2,
        "false_rollback": {k: {"count": v, "rate": r4(v / n_seeds)} for k, v in same.items()},
        "detect_bad": {k: {"count": v, "rate": r4(v / n_seeds)} for k, v in bad.items()},
        "world_shift": {"n_seeds": shift_seeds, "shift": 1.0,
                        "false_rollback": {k: {"count": v, "rate": r4(v / shift_seeds)} for k, v in world.items()},
                        "v1_error_before": r4(np.mean(v1_err["before"])),
                        "v1_error_after_shift": r4(np.mean(v1_err["after_shift"]))},
    }


def e5(v1, v2):
    shared = D.serve_view(D.make_table(9000, 5000), skew=False)
    out = {mode: {k: r4(v) for k, v in SC.rollback_accuracy(v1, v2, shared, mode).items()} for mode in ("in_place", "expand")}
    out["v1_predicts_churn_after_in_place"] = r4(M.predict(v1, SC.migrate(shared, "in_place")).mean())
    out["churn_rate"] = r4(shared["churned"].mean())
    return out


def e6(tr, v1, v2):
    _, _, t1, t2 = train_models(fit=MT.fit_logistic)
    T = traffic(0, v1, v2)
    tf = D.serve_view(D.make_table(1000, N_REQ))
    c1t, c2t = M.correct(t1, tf), M.correct(t2, tf)
    r_np = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "canary", DEFAULT_GUARD, p=0.05, label_delay=1000)
    r_t = simulate_traffic(c1t, c2t, T["users"], T["seg"], "canary", DEFAULT_GUARD, p=0.05, label_delay=1000)
    return {
        "max_abs_w_diff": float(max(np.max(np.abs(v1["w"] - t1["w"])), np.max(np.abs(v2["w"] - t2["w"])))),
        "max_abs_b_diff": float(max(abs(v1["b"] - t1["b"]), abs(v2["b"] - t2["b"]))),
        "pred_mismatch": int(np.sum(M.predict(v1, tf) != M.predict(t1, tf)) + np.sum(M.predict(v2, tf) != M.predict(t2, tf))),
        "canary_0.05_L1000_numpy": {k: r_np[k] for k in ("decision", "decided_at", "harm")},
        "canary_0.05_L1000_torch": {k: r_t[k] for k in ("decision", "decided_at", "harm")},
    }


def e7():
    n_users, per_user = 2000, 5
    users = [f"user-{i}" for i in range(n_users)]
    a = [route(u, 0.05) for u in users]
    b = [route(u, 0.2) for u in users]
    rng = np.random.default_rng(0)
    both = 0
    for u in users:
        seen = {route_random(rng, 0.2) for _ in range(per_user)}
        both += len(seen) == 2
    hash_both = sum(len({route(u, 0.2) for _ in range(per_user)}) == 2 for u in users)
    share = float(np.mean([bucket(f"x{i}") < 0.2 for i in range(100_000)]))
    return {
        "n_users": n_users, "requests_per_user": per_user,
        "raise_0.05_to_0.2": {"v2_to_v1": sum(x == "v2" and y == "v1" for x, y in zip(a, b)),
                              "v1_to_v2": sum(x == "v1" and y == "v2" for x, y in zip(a, b)),
                              "v2_at_0.05": a.count("v2"), "v2_at_0.2": b.count("v2")},
        "users_seeing_both_random": both, "users_seeing_both_random_rate": r4(both / n_users),
        "users_seeing_both_hash": hash_both,
        "hash_share_at_0.2_of_100k": r4(share),
        "random_expected_rate": r4(1 - 0.8**per_user - 0.2**per_user),
    }


def main():
    t0 = time.time()
    tr, va, v1, v2 = train_models()
    trafs = {s: traffic(s, v1, v2) for s in SEEDS}
    res = {
        "env": {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
                "numpy": np.__version__, "torch": torch.__version__},
        "checked": "2026-10-08", "threads": 2, "n_requests": N_REQ, "seeds": list(SEEDS),
        "guard": DEFAULT_GUARD, "check_every": 100,
        "E1": e1(tr, va, v1, v2),
        "E2": e2(trafs),
        "E3": e3(trafs),
        "E4": e4(v1, v2),
        "E5": e5(v1, v2),
        "E6": e6(tr, v1, v2),
        "E7": e7(),
    }
    res["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E1", "E5", "E6", "E7")}, ensure_ascii=False, indent=1))
    print("seconds", res["seconds"])


if __name__ == "__main__":
    main()
