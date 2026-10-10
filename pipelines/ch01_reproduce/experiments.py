"""1장 비교 실험. `uv run python pipelines/ch01_reproduce/experiments.py` 로 실행한다.

E1 시드 없이 5번: 같은 코드, 같은 데이터인데 정확도가 몇 개 나오나
E2 시드 없이 20번 vs 시드 0으로 20번: 표준편차
E3 시드 0~9 각각: 평균 ± 표준편차(보고 기준), 운 좋은 시드와 운 나쁜 시드의 차이
E4 무작위의 출처 나누기: 분할만 바꿀 때 vs 초기값·순서만 바꿀 때
E5 흔들리는 테스트: "정확도 0.84 이상" 테스트가 시드 없는 실행에서 몇 번 통과하나
   (별도 키 없이 E2 결과의 unseeded_pass에 들어 있다)
E6 기록해 둔 엔트로피로 시드 없던 실행 다시 내기
E7 부동소수 합의 순서: float32 10^6개를 원래 순서/섞은 순서, 한 개씩/나눠서/NumPy/정확한 합
E8 같은 시드, 다른 계산 환경: float64 vs float32, NumPy vs PyTorch, PyTorch 스레드 1 vs 2
E9 환경 기록과 다른 컴퓨터 결과 비교 형식, 기록에 드는 시간
E10 연습 1의 정답: 학습률 0.05 vs 0.1을 같은 분할에서 비교 vs 다른 분할에서 비교

시드 없는 실행(E1, E2, E5, E6)의 숫자는 실행할 때마다 달라진다. 받은 엔트로피를 같이 저장해 두었으니
그 값을 cfg["entropy"]로 넘기면 같은 숫자를 다시 낼 수 있다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch01_model_np as mnp  # noqa: E402
import pipelines_ch01_model_torch as mtt  # noqa: E402
import pipelines_ch01_repro as rp  # noqa: E402

torch.set_num_threads(2)

HERE = Path(__file__).parent
OUT = HERE / "results" / "ch01.json"
PASS_LINE = 0.84


def _stats(accs):
    return {
        "n": len(accs),
        "mean": statistics.fmean(accs),
        "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "min": min(accs),
        "max": max(accs),
        "range": max(accs) - min(accs),
        "distinct": len(set(accs)),
    }


def _public(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def e1_unseeded_five():
    runs = [_public(mnp.train({"seed": None})) for _ in range(5)]
    return {"runs": runs, "stats": _stats([r["accuracy"] for r in runs])}


def e2_unseeded_vs_seeded():
    unseeded = [_public(mnp.train({"seed": None})) for _ in range(20)]
    seeded = [_public(mnp.train({"seed": 0})) for _ in range(20)]
    return {
        "unseeded": {"stats": _stats([r["accuracy"] for r in unseeded]),
                     "accuracies": [r["accuracy"] for r in unseeded],
                     "entropies": [r["entropy"] for r in unseeded]},
        "seeded_0": {"stats": _stats([r["accuracy"] for r in seeded]),
                     "distinct_weight_fps": len({r["weights_fp"] for r in seeded}),
                     "weights_fp": seeded[0]["weights_fp"]},
        "pass_line": PASS_LINE,
        "unseeded_pass": sum(r["accuracy"] >= PASS_LINE for r in unseeded),
    }


def e3_seed_sweep():
    rows = [{"seed": s, **{k: v for k, v in _public(mnp.train({"seed": s})).items() if k in ("accuracy", "correct")}}
            for s in range(10)]
    accs = [r["accuracy"] for r in rows]
    best = max(rows, key=lambda r: r["accuracy"])
    worst = min(rows, key=lambda r: r["accuracy"])
    return {"rows": rows, "stats": _stats(accs), "best": best, "worst": worst,
            "pass_line": PASS_LINE, "seeds_passing": [r["seed"] for r in rows if r["accuracy"] >= PASS_LINE]}


def e4_sources():
    split_only = [mnp.train({"split_seed": s, "train_seed": 0})["accuracy"] for s in range(20)]
    train_only = [mnp.train({"split_seed": 0, "train_seed": s})["accuracy"] for s in range(20)]
    return {"split_only": _stats(split_only), "train_only": _stats(train_only)}


def e6_replay():
    first = mnp.train({"seed": None})
    again = mnp.train({"entropy": int(first["entropy"])})
    return {"entropy": first["entropy"], "accuracy": first["accuracy"], "replay_accuracy": again["accuracy"],
            "weights_equal": bool(np.array_equal(first["_w"], again["_w"]) and first["_b"] == again["_b"])}


def e7_float_sum():
    x = np.random.default_rng(0).uniform(0.0, 1.0, 10**6).astype(np.float32)
    xs = x[np.random.default_rng(1).permutation(len(x))]
    exact = rp.exact_sum(x)
    out = {"n": len(x), "exact_fsum": exact}
    for name, arr in (("original", x), ("shuffled", xs)):
        out[name] = {
            "np_sum_float32": float(np.sum(arr)),
            "sequential_float32": float(rp.sequential_sum(arr)),
            "np_sum_float64": float(np.sum(arr.astype(np.float64))),
            "chunked_float32": {str(k): float(rp.chunked_sum(arr, k)) for k in (1, 2, 4, 8, 16)},
        }
    out["np_sum_diff_original_vs_shuffled"] = out["original"]["np_sum_float32"] - out["shuffled"]["np_sum_float32"]
    out["sequential_diff_original_vs_shuffled"] = (out["original"]["sequential_float32"]
                                                   - out["shuffled"]["sequential_float32"])
    out["chunked_distinct_values_original"] = len(set(out["original"]["chunked_float32"].values()))
    a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
    out["hand"] = {
        "float32_(1e8+-1e8)+1": float((a + b) + c),
        "float32_1e8+(-1e8+1)": float(a + (b + c)),
        "float32_-1e8+1": float(b + c),
        "float64_(0.1+0.2)+0.3": repr((0.1 + 0.2) + 0.3),
        "float64_0.1+(0.2+0.3)": repr(0.1 + (0.2 + 0.3)),
    }
    return out


def e8_environment():
    base = mnp.prepare({"seed": 0})
    c = base["cfg"]
    w64, b64 = mnp.fit_sgd(base["X_tr"], base["y_tr"], base["w0"], base["b0"], base["orders"], c["batch"], c["lr"])
    p64 = mnp.predict_proba(base["X_te"], w64, b64)
    f32 = mnp.prepare({"seed": 0, "dtype": "float32"})
    w32, b32 = mnp.fit_sgd(f32["X_tr"], f32["y_tr"], f32["w0"], f32["b0"], f32["orders"], c["batch"], np.float32(c["lr"]))
    p32 = mnp.predict_proba(f32["X_te"], w32, b32)
    tw64, tb64 = mtt.fit_sgd(base["X_tr"], base["y_tr"], base["w0"], base["b0"], base["orders"], c["batch"], c["lr"])
    tp64 = mnp.predict_proba(base["X_te"], tw64, tb64)
    tw32, tb32 = mtt.fit_sgd(f32["X_tr"], f32["y_tr"], f32["w0"], f32["b0"], f32["orders"], c["batch"], c["lr"])
    tp32 = mnp.predict_proba(f32["X_te"], tw32, np.float32(tb32))
    y = base["y_te"]

    def acc(p):
        return float(np.mean((p >= 0.5) == y))

    def flips(p, q):
        return int(np.sum((p >= 0.5) != (q >= 0.5)))

    # 같은 float32 배열의 합: NumPy vs PyTorch, PyTorch 스레드 1 vs 2
    x = np.random.default_rng(0).uniform(0.0, 1.0, 10**6).astype(np.float32)
    tx = torch.from_numpy(x)
    sums = {}
    for th in (1, 2):
        torch.set_num_threads(th)
        sums[f"torch_threads_{th}"] = float(tx.sum())
    torch.set_num_threads(2)
    sums["numpy"] = float(np.sum(x))
    sums["exact_fsum"] = rp.exact_sum(x)
    # 같은 float32 행렬곱: 스레드 1 vs 2
    A = torch.from_numpy(np.random.default_rng(2).normal(size=(512, 512)).astype(np.float32))
    mm = {}
    for th in (1, 2):
        torch.set_num_threads(th)
        mm[th] = (A @ A).numpy().copy()
    torch.set_num_threads(2)
    det_ok = True
    try:
        torch.use_deterministic_algorithms(True)
        _ = A @ A
    except RuntimeError:
        det_ok = False
    finally:
        torch.use_deterministic_algorithms(False)
    return {
        "numpy_float64": {"accuracy": acc(p64), "weights_fp": mnp.fingerprint(w64, np.asarray(b64))},
        "numpy_float32": {"accuracy": acc(p32), "max_abs_weight_diff_vs_f64": float(np.max(np.abs(w32 - w64))),
                          "max_abs_proba_diff_vs_f64": float(np.max(np.abs(p32 - p64))), "flips_vs_f64": flips(p32, p64)},
        "torch_float64": {"accuracy": acc(tp64), "max_abs_weight_diff_vs_np": float(np.max(np.abs(tw64 - w64))),
                          "max_abs_proba_diff_vs_np": float(np.max(np.abs(tp64 - p64))), "flips_vs_np": flips(tp64, p64)},
        "torch_float32": {"accuracy": acc(tp32), "max_abs_weight_diff_vs_np32": float(np.max(np.abs(tw32 - w32))),
                          "flips_vs_np32": flips(tp32, p32)},
        "sum_1e6_float32": sums,
        "sum_numpy_minus_torch": sums["numpy"] - sums["torch_threads_2"],
        "sum_torch_threads_bitwise_equal": sums["torch_threads_1"] == sums["torch_threads_2"],
        "matmul_512_threads_bitwise_equal": bool(np.array_equal(mm[1], mm[2])),
        "matmul_512_threads_max_abs_diff": float(np.max(np.abs(mm[1] - mm[2]))),
        "use_deterministic_algorithms_cpu_matmul_ok": det_ok,
    }


def e9_env_and_cost():
    t0 = time.perf_counter()
    for _ in range(100):
        rp.set_seeds(0)
    t_seed = (time.perf_counter() - t0) / 100
    t0 = time.perf_counter()
    for _ in range(100):
        snap = rp.env_snapshot()
    t_snap = (time.perf_counter() - t0) / 100
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        mnp.train({"seed": 0})
        times.append(time.perf_counter() - t0)
    m = mnp.train({"seed": 0})
    return {
        "env": snap,
        "keys": rp.ENV_KEYS,
        "set_seeds_ms": t_seed * 1e3,
        "env_snapshot_ms": t_snap * 1e3,
        "train_ms_median": statistics.median(times) * 1e3,
        "compare_record": {"seed": 0, "accuracy": m["accuracy"], "weights_fp": m["weights_fp"],
                           "proba_fp": m["proba_fp"], "env": snap},
    }


def e10_exercise_lr():
    """연습 1 정답. train_seed는 0으로 고정, 분할 시드 s=0..9.
    같은 분할: 두 모델 모두 split_seed=s. 다른 분할: 학습률 0.05 모델만 split_seed=s+10."""
    base = [mnp.train({"split_seed": s, "train_seed": 0})["accuracy"] for s in range(10)]
    same = [mnp.train({"split_seed": s, "train_seed": 0, "lr": 0.05})["accuracy"] - base[s] for s in range(10)]
    diff = [mnp.train({"split_seed": s + 10, "train_seed": 0, "lr": 0.05})["accuracy"] - base[s] for s in range(10)]
    def st(d):
        return {"min": min(d), "max": max(d), "mean": statistics.fmean(d), "diffs": d}
    return {"same_split": st(same), "different_split": st(diff)}


def main():
    t0 = time.perf_counter()
    res = {
        "chapter": "pipelines/ch01_reproduce",
        "note": "E1, E2, E5, E6의 시드 없는 실행은 실행마다 달라진다. entropies로 다시 낼 수 있다.",
        "cfg": {k: v for k, v in mnp.DEFAULT_CFG.items()},
        "data": {"n_rows": 2000, "n_test": 400, "data_seed": 2026, "accuracy_step": 1 / 400},
        "E1_unseeded_five": e1_unseeded_five(),
        "E2_unseeded_vs_seeded": e2_unseeded_vs_seeded(),
        "E3_seed_sweep": e3_seed_sweep(),
        "E4_sources": e4_sources(),
        "E6_replay": e6_replay(),
        "E7_float_sum": e7_float_sum(),
        "E8_environment": e8_environment(),
        "E9_env_and_cost": e9_env_and_cost(),
        "E10_exercise_lr": e10_exercise_lr(),
    }
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: res[k] for k in ("E1_unseeded_five", "E3_seed_sweep")}, ensure_ascii=False)[:1500])
    print("saved", OUT, f"{res['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
