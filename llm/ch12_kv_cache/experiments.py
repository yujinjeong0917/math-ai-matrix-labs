"""12장 비교 실험. `uv run python llm/ch12_kv_cache/experiments.py` 로 실행한다.

모델(생성 비교용, 무작위 가중치): 어휘 64, d=64, 층 2개, 헤드 4개(dk=16), 위치 표 513칸. 프롬프트는 토큰 1개.
E1 같은 결과: 캐시 있음/없음의 로짓 차이, 토큰열 일치, 앞 위치 k·v 불변(2층), 마스크를 뺀 대조
E2 실패 재현과 비교(NumPy): 생성 길이 {64,128,256,512}에서 캐시 없음 / 캐시(MHA) / 캐시(MQA)
    이론 FLOP, 순전파한 위치 수, 실측 시간(7회 중앙값), 캐시 메모리, 스텝별 시간
E3 PyTorch 캐시 두 방식: torch.cat으로 늘리기 vs 미리 잡고 포인터 옮기기 (float32, 단일 스레드)
E4 품질: MHA(h_kv=4) / GQA(h_kv=2) / MQA(h_kv=1)를 같은 과제로 새로 학습, 시드 3개
E5 흔한 실패: 위치 번호를 잘못 준 캐시, 프롬프트가 바뀌었는데 비우지 않은 캐시

결과는 results/ch12.json 에 저장한다. 실측 시간은 기기마다 다르지만 FLOP, 위치 수, 메모리, 테스트 결과는 같아야 한다.
"""

import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import kvcache_np as kv
import kvcache_torch as kt

OUT = Path(__file__).parent / "results" / "ch12.json"
GEN = dict(vocab=64, d=64, L=2, h=4)
LENGTHS = (64, 128, 256, 512)
REPEATS = 7
torch.set_num_threads(1)


def model(h_kv=4, seed=0):
    return kv.init_params(GEN["vocab"], 513, GEN["d"], GEN["L"], GEN["h"], h_kv=h_kv, seed=seed)


# ---------------------------------------------------------------- E1


def e1_equivalence():
    p = model()
    sa, la = kv.generate_no_cache(p, [0], 64, record_logits=True)
    sb, lb = kv.generate_with_cache(p, [0], 64, record_logits=True)
    ra = kv.generate_no_cache(p, [0], 64, rng=np.random.default_rng(42))
    rb = kv.generate_with_cache(p, [0], 64, rng=np.random.default_rng(42))
    toks = np.random.default_rng(1).integers(0, 64, 40)
    _, short = kv.forward(p, toks[:20], return_kv=True)
    _, long = kv.forward(p, toks, return_kv=True)
    _, short_nm = kv.forward(p, toks[:20], causal=False, return_kv=True)
    _, long_nm = kv.forward(p, toks, causal=False, return_kv=True)
    return {
        "steps": 64,
        "max_logit_diff_float64": float(max(np.abs(a - b).max() for a, b in zip(la, lb))),
        "greedy_tokens_equal": sa == sb,
        "sampled_tokens_equal_seed42": ra == rb,
        "layer2_prefix_k_max_change_causal": float(np.abs(long[1][0][:20] - short[1][0]).max()),
        "layer2_prefix_k_max_change_no_mask": float(np.abs(long_nm[1][0][:20] - short_nm[1][0]).max()),
        "layer2_prefix_k_typical_size": float(np.abs(short[1][0]).mean()),
    }


# ---------------------------------------------------------------- E2


def _timed_generate(p, T, cached):
    """kv.generate_*와 같은 루프에 스텝별 타이머만 붙였다."""
    seq, step_t = [0], []
    if cached:
        cache = kv.new_cache(p, T + 1)
        t0 = time.perf_counter()
        logits = kv.forward(p, seq, 0, cache, last_only=True)
        step_t.append(time.perf_counter() - t0)
        for i in range(T):
            seq.append(int(np.argmax(logits)))
            if i < T - 1:
                t0 = time.perf_counter()
                logits = kv.forward(p, seq[-1:], len(seq) - 1, cache, last_only=True)
                step_t.append(time.perf_counter() - t0)
    else:
        for _ in range(T):
            t0 = time.perf_counter()
            logits = kv.forward(p, seq, 0, last_only=True)
            step_t.append(time.perf_counter() - t0)
            seq.append(int(np.argmax(logits)))
    return seq, step_t


def e2_generation_cost():
    models = {"no_cache": (model(4), False), "cache_mha": (model(4), True), "cache_mqa": (model(1, seed=1), True)}
    c = model()["cfg"]
    rows = []
    step_curve = {}
    for T in LENGTHS:
        row = {"T": T}
        seqs = {}
        for name, (p, cached) in models.items():
            h_kv = p["cfg"]["h_kv"]
            args = dict(d=c["d"], L=c["L"], h=c["h"], h_kv=h_kv, dk=c["dk"], vocab=c["vocab"])
            fl = (kv.flops_with_cache if cached else kv.flops_no_cache)(1, T, **args)
            totals, steps = [], []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                seq, st = _timed_generate(p, T, cached)
                totals.append(time.perf_counter() - t0)
                steps.append(st)
            row[name] = {
                "flops": int(fl),
                "positions_per_layer": kv.positions_processed(1, T, cached),
                "ms_median": float(np.median(totals) * 1e3),
                "cache_bytes_float64": kv.cache_bytes(c["L"], T, h_kv, c["dk"], 8) if cached else 0,
            }
            seqs[name] = seq
            if T == LENGTHS[-1]:
                med = np.median(np.array(steps), axis=0)  # 스텝별로 7회 중앙값
                # 한 스텝은 0.1ms 안팎이라 잡음이 크다. 스텝 s 앞의 16스텝 평균으로 적는다(s=1은 그 스텝만).
                step_curve[name] = {str(s): float(med[max(0, s - 16): s].mean() * 1e3) for s in (1, 64, 128, 256, 512)}
        row["tokens_equal_no_cache_vs_mha"] = seqs["no_cache"] == seqs["cache_mha"]
        row["flop_ratio_no_cache_over_mha"] = row["no_cache"]["flops"] / row["cache_mha"]["flops"]
        row["time_ratio_no_cache_over_mha"] = row["no_cache"]["ms_median"] / row["cache_mha"]["ms_median"]
        rows.append(row)
    slope = {}
    for name in models:
        t = [r[name]["ms_median"] for r in rows]
        slope[name] = float(np.polyfit(np.log(LENGTHS), np.log(t), 1)[0])
        slope[name + "_256_to_512_ratio"] = t[3] / t[2]
        f = [r[name]["flops"] for r in rows]
        slope[name + "_flops"] = float(np.polyfit(np.log(LENGTHS), np.log(f), 1)[0])
        slope[name + "_flops_256_to_512_ratio"] = f[3] / f[2]
    c = model()["cfg"]
    per_step = lambda t: c["L"] * kv.layer_flops(t, t, c["d"], c["h"], c["h"], c["dk"]) + 2 * c["d"] * c["vocab"]  # noqa: E731
    slope["no_cache_step_flops_512_over_256"] = per_step(512) / per_step(256)
    return {"rows": rows, "step_ms_at_T512": step_curve, "loglog_slope_time_vs_T": slope,
            "flop_note": "행렬곱만 센다(곱셈+덧셈=2). LN, softmax, 임베딩 조회, 잔차 덧셈 제외. 출력층은 스텝마다 마지막 위치 하나."}


# ---------------------------------------------------------------- E3


def e3_torch_caches():
    out = {}
    p = model()
    m = kt.from_numpy(p).float()
    T = 512
    for B in (1, 16):
        prompt = torch.zeros(B, 1, dtype=torch.long)
        res = {}
        seqs = {}
        for mode in ("cat", "prealloc"):
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                seqs[mode] = kt.generate(m, prompt, T, mode)
                times.append(time.perf_counter() - t0)
            res[mode + "_ms"] = float(np.median(times) * 1e3)
        res["same_tokens"] = bool(torch.equal(seqs["cat"], seqs["prealloc"]))
        c = p["cfg"]
        res["prealloc_reserved_bytes_float32"] = kv.cache_bytes(c["L"], T + 1, c["h_kv"], c["dk"], 4) * B
        out[f"B{B}"] = res
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        kt.generate(m, torch.zeros(1, 1, dtype=torch.long), T, "none")
        times.append(time.perf_counter() - t0)
    out["B1"]["none_ms_median3"] = float(np.median(times) * 1e3)
    return out


# ---------------------------------------------------------------- E4
# 데이터는 9장과 같은 2차 규칙 말뭉치다. 출처: llm/ch09_residual_norm/experiments.py (복사)

V4, N4, P_RULE = 16, 16, 0.8


def make_corpus(n_seq, seed):
    rng = np.random.default_rng(seed)
    T = np.random.default_rng(12345).integers(0, V4, (V4, V4))
    X = np.empty((n_seq, N4), dtype=np.int64)
    X[:, :2] = rng.integers(0, V4, (n_seq, 2))
    for t in range(2, N4):
        rule = T[X[:, t - 2], X[:, t - 1]]
        X[:, t] = np.where(rng.random(n_seq) < P_RULE, rule, rng.integers(0, V4, n_seq))
    return torch.tensor(X)


def entropy_floor():
    p_hit = P_RULE + (1 - P_RULE) / V4
    p_miss = (1 - P_RULE) / V4
    h_rule = -(p_hit * math.log(p_hit) + (V4 - 1) * p_miss * math.log(p_miss))
    return (math.log(V4) + (N4 - 2) * h_rule) / (N4 - 1)


def e4_quality(steps=1500, batch=64, lr=3e-3):
    train, val = make_corpus(100_000, 0), make_corpus(2000, 1)  # 학습 표본을 넉넉히 둬서 과적합이 아니라 구조 차이를 본다
    out = {"entropy_floor": entropy_floor(), "steps": steps, "batch": batch, "lr": lr, "d": 32, "L": 2, "h": 4}
    for name, h_kv in (("MHA", 4), ("GQA", 2), ("MQA", 1)):
        losses = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            m = kt.TinyGPT(V4, N4, 32, 2, 4, h_kv)
            opt = torch.optim.AdamW(m.parameters(), lr=lr)
            g = torch.Generator().manual_seed(seed)
            for _ in range(steps):
                xb = train[torch.randint(0, len(train), (batch,), generator=g)]
                loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, V4), xb[:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                vl = F.cross_entropy(m(val[:, :-1]).reshape(-1, V4), val[:, 1:].reshape(-1))
            losses.append(float(vl))
        out[name] = {
            "h_kv": h_kv,
            "params": sum(p.numel() for p in m.parameters()),
            "val_loss_by_seed": losses,
            "val_loss_mean": float(np.mean(losses)),
            "cache_floats_per_token": 2 * 2 * h_kv * 8,  # K·V × 층 2 × h_kv × dk 8
        }
    return out


# ---------------------------------------------------------------- E5


def e5_bugs():
    p = model()
    ref_seq, ref_logits = kv.generate_with_cache(p, [0], 64, record_logits=True)

    # 버그 1: 새 토큰을 넣을 때 위치를 늘 0으로 준다(start를 빼먹음)
    cache = kv.new_cache(p, 65)
    seq, logs = [0], []
    logits = kv.forward(p, seq, 0, cache, last_only=True)
    for i in range(64):
        logs.append(logits)
        seq.append(int(np.argmax(logits)))
        if i < 63:
            logits = kv.forward(p, seq[-1:], 0, cache, last_only=True)
    diff1 = [float(np.abs(a - b).max()) for a, b in zip(logs, ref_logits)]
    first_bad = next((i for i, (a, b) in enumerate(zip(seq, ref_seq)) if a != b), None)

    # 버그 2: 다른 프롬프트로 넘어가면서 캐시를 비우지 않는다
    cache = kv.new_cache(p, 200)
    kv.generate_with_cache(p, [5, 6, 7], 20, cache=cache)
    stale_seq, stale_logits = kv.generate_with_cache(p, [0], 64, record_logits=True, cache=cache)
    diff2 = float(np.abs(stale_logits[0] - ref_logits[0]).max())
    return {
        "wrong_position": {"step1_logit_diff": diff1[0], "step2_logit_diff": diff1[1], "max_logit_diff": max(diff1),
                           "first_token_mismatch_index": first_bad},
        "stale_cache": {"first_step_logit_diff": diff2,
                        "tokens_equal": stale_seq == ref_seq},
    }


def main():
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "torch_threads": torch.get_num_threads(), "repeats": REPEATS},
        "model_for_generation": {**GEN, "dk": GEN["d"] // GEN["h"], "ff": 4, "prompt_len": 1},
    }
    for name, fn in (("E1", e1_equivalence), ("E2", e2_generation_cost), ("E3", e3_torch_caches),
                     ("E4", e4_quality), ("E5", e5_bugs)):
        t0 = time.perf_counter()
        res[name] = fn()
        print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
