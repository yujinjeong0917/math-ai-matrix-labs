"""12장 비교 실험. `uv run python llm/ch12_kv_cache/experiments.py` 로 실행한다.

모델(생성 비교용, 무작위 가중치): 어휘 64, d=64, 층 2개, 헤드 4개(dk=16), 위치 표 513칸. 프롬프트는 토큰 1개.
E1 같은 결과: 캐시 있음/없음의 로짓 차이, 토큰열 일치, 앞 위치 k·v 불변(2층), 마스크를 뺀 대조
E2 실패 재현과 비교(NumPy): 생성 길이 {64,128,256,512}에서 캐시 없음 / 캐시(MHA) / 캐시(MQA)
    이론 FLOP, 순전파한 위치 수, 실측 시간(7회 중앙값), 캐시 메모리, 스텝별 시간
E3 PyTorch 캐시 두 방식: torch.cat으로 늘리기 vs 미리 잡고 포인터 옮기기 (float32, 단일 스레드)
E4 품질: MHA(h_kv=4) / GQA(h_kv=2) / MQA(h_kv=1)를 같은 과제로 새로 학습, 시드 3개
E5 흔한 실패: 위치 번호를 잘못 준 캐시, 프롬프트가 바뀌었는데 비우지 않은 캐시

결과는 results/ch12.json 에 저장한다. 실측 시간은 기기마다 다르지만 FLOP, 위치 수, 메모리, 테스트 결과는 같아야 한다.

입력: 없음(시드는 파일 안에 고정). 출력: results/ch12.json. 단위: 시간 ms, 메모리 바이트, 연산량 FLOP(곱셈+덧셈 = 2).
모델 가중치는 무작위(학습하지 않음): 캐시 유무 비교는 값이 같고 비용만 다르다는 것을 보이는 실험이라 학습이 필요 없다. E4만 새로 학습한다.
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
# import 시점에 단일 스레드로 고정(시간 비교의 재현성). 이 모듈을 import만 해도 전역 설정이 바뀐다(주의)
torch.set_num_threads(1)


def model(h_kv=4, seed=0):
    # 위치 표 513칸 = 최대 생성 512 + 프롬프트 1. h_kv로 MHA(4)/GQA(2)/MQA(1) 선택
    return kv.init_params(GEN["vocab"], 513, GEN["d"], GEN["L"], GEN["h"], h_kv=h_kv, seed=seed)


# ---------------------------------------------------------------- E1


def e1_equivalence():
    p = model()
    # 캐시 없음 vs 캐시 있음 탐욕 생성 64스텝. la, lb는 스텝별 로짓
    sa, la = kv.generate_no_cache(p, [0], 64, record_logits=True)
    sb, lb = kv.generate_with_cache(p, [0], 64, record_logits=True)
    # 같은 시드(42)로 표집해도 토큰이 같은지
    ra = kv.generate_no_cache(p, [0], 64, rng=np.random.default_rng(42))
    rb = kv.generate_with_cache(p, [0], 64, rng=np.random.default_rng(42))
    toks = np.random.default_rng(1).integers(0, 64, 40)
    # 앞 20토큰만 넣은 경우와 40토큰 전체의 층별 k, v
    _, short = kv.forward(p, toks[:20], return_kv=True)
    _, long = kv.forward(p, toks, return_kv=True)
    _, short_nm = kv.forward(p, toks[:20], causal=False, return_kv=True)
    _, long_nm = kv.forward(p, toks, causal=False, return_kv=True)
    return {
        "steps": 64,
        "max_logit_diff_float64": float(max(np.abs(a - b).max() for a, b in zip(la, lb))),
        "greedy_tokens_equal": sa == sb,
        "sampled_tokens_equal_seed42": ra == rb,
        # [1] = 2층, [0] = k. causal이면 변화가 0에 가까워야 한다
        "layer2_prefix_k_max_change_causal": float(np.abs(long[1][0][:20] - short[1][0]).max()),
        # 마스크가 없으면 2층 k가 뒤 토큰 때문에 바뀐다(대조)
        "layer2_prefix_k_max_change_no_mask": float(np.abs(long_nm[1][0][:20] - short_nm[1][0]).max()),
        # 변화량을 비교할 기준 크기(k 절댓값 평균)
        "layer2_prefix_k_typical_size": float(np.abs(short[1][0]).mean()),
    }


# ---------------------------------------------------------------- E2


def _timed_generate(p, T, cached):
    """kv.generate_*와 같은 루프에 스텝별 타이머만 붙였다."""
    # 프롬프트는 토큰 0 하나. step_t[i] = i번째 순전파 시간(초)
    seq, step_t = [0], []
    if cached:
        cache = kv.new_cache(p, T + 1)
        t0 = time.perf_counter()
        # 프리필(프롬프트 1개)도 step_t에 첫 칸으로 기록된다
        logits = kv.forward(p, seq, 0, cache, last_only=True)
        step_t.append(time.perf_counter() - t0)
        for i in range(T):
            seq.append(int(np.argmax(logits)))
            # 마지막 토큰은 순전파를 생략(kv.generate_with_cache와 같은 규칙)
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
    # cache_mqa만 시드 1: 가중치가 달라 토큰열이 MHA와 같을 이유가 없다. 이 비교는 시간·메모리 비교용이고, 토큰 일치 검사는 no_cache vs cache_mha만 한다
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
            # 이론 FLOP. 프롬프트 길이 1, 생성 T
            fl = (kv.flops_with_cache if cached else kv.flops_no_cache)(1, T, **args)
            totals, steps = [], []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                seq, st = _timed_generate(p, T, cached)
                totals.append(time.perf_counter() - t0)
                steps.append(st)
            # flops와 positions는 이론값(결정적), ms_median은 실측(기기마다 다름)
            row[name] = {
                "flops": int(fl),
                "positions_per_layer": kv.positions_processed(1, T, cached),
                "ms_median": float(np.median(totals) * 1e3),
                # 채워진 캐시 크기 공식 2*L*T*h_kv*dk*8. 캐시 없음은 0
                "cache_bytes_float64": kv.cache_bytes(c["L"], T, h_kv, c["dk"], 8) if cached else 0,
            }
            seqs[name] = seq
            if T == LENGTHS[-1]:
                # steps: (REPEATS, 스텝 수) -> 스텝별 중앙값
                med = np.median(np.array(steps), axis=0)  # 스텝별로 7회 중앙값
                # 한 스텝은 0.1ms 안팎이라 잡음이 크다. 스텝 s 앞의 16스텝 평균으로 적는다(s=1은 그 스텝만).
                # 스텝 s 앞 16스텝을 평균한 값(med[s-16:s]). s = 1이면 med[0:1]로 프리필 한 스텝뿐
                step_curve[name] = {str(s): float(med[max(0, s - 16): s].mean() * 1e3) for s in (1, 64, 128, 256, 512)}
        row["tokens_equal_no_cache_vs_mha"] = seqs["no_cache"] == seqs["cache_mha"]
        row["flop_ratio_no_cache_over_mha"] = row["no_cache"]["flops"] / row["cache_mha"]["flops"]
        row["time_ratio_no_cache_over_mha"] = row["no_cache"]["ms_median"] / row["cache_mha"]["ms_median"]
        rows.append(row)
    slope = {}
    for name in models:
        t = [r[name]["ms_median"] for r in rows]
        # 로그-로그 직선의 기울기 = 시간 ~ T^기울기의 지수. 캐시 없음은 2~3에 가까울 것으로 기대(실측이라 확인 필요), 캐시는 약 1~2
        slope[name] = float(np.polyfit(np.log(LENGTHS), np.log(t), 1)[0])
        slope[name + "_256_to_512_ratio"] = t[3] / t[2]
        f = [r[name]["flops"] for r in rows]
        slope[name + "_flops"] = float(np.polyfit(np.log(LENGTHS), np.log(f), 1)[0])
        slope[name + "_flops_256_to_512_ratio"] = f[3] / f[2]
    # E2 루프 변수 c를 같은 값으로 다시 만든다
    c = model()["cfg"]
    # 캐시 없음에서 접두사 길이 t일 때 한 스텝의 FLOP(MHA 기준 h_kv = h). 512/256 배율은 t에 대한 증가율
    per_step = lambda t: c["L"] * kv.layer_flops(t, t, c["d"], c["h"], c["h"], c["dk"]) + 2 * c["d"] * c["vocab"]  # noqa: E731
    slope["no_cache_step_flops_512_over_256"] = per_step(512) / per_step(256)
    return {"rows": rows, "step_ms_at_T512": step_curve, "loglog_slope_time_vs_T": slope,
            "flop_note": "행렬곱만 센다(곱셈+덧셈=2). LN, softmax, 임베딩 조회, 잔차 덧셈 제외. 출력층은 스텝마다 마지막 위치 하나."}


# ---------------------------------------------------------------- E3


def e3_torch_caches():
    out = {}
    p = model()
    # float64 -> float32: 실제 쓰이는 정밀도에 맞춘 시간 측정
    m = kt.from_numpy(p).float()
    T = 512
    for B in (1, 16):
        # 프롬프트는 토큰 0 하나, 배치 B = 1 또는 16
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
        # 미리 잡은 크기(T+1칸)를 배치 B배: 쓰지 않은 칸도 포함한 예약 메모리
        res["prealloc_reserved_bytes_float32"] = kv.cache_bytes(c["L"], T + 1, c["h_kv"], c["dk"], 4) * B
        out[f"B{B}"] = res
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        # 캐시 없음은 느려서 3회만 측정(none_ms_median3)
        kt.generate(m, torch.zeros(1, 1, dtype=torch.long), T, "none")
        times.append(time.perf_counter() - t0)
    out["B1"]["none_ms_median3"] = float(np.median(times) * 1e3)
    return out


# ---------------------------------------------------------------- E4
# 데이터는 9장과 같은 2차 규칙 말뭉치다. 출처: llm/ch09_residual_norm/experiments.py (복사)

# 9장과 같은 규칙 말뭉치 설정: 어휘 16, 길이 16, 규칙 확률 0.8
V4, N4, P_RULE = 16, 16, 0.8


def make_corpus(n_seq, seed):
    rng = np.random.default_rng(seed)
    # 규칙 표는 모든 실험에서 같도록 시드 고정(9장과 같은 값)
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
    # 위치 1은 균등(log V), 나머지 N4 - 2개는 규칙 엔트로피. 예측 대상은 N4 - 1개
    return (math.log(V4) + (N4 - 2) * h_rule) / (N4 - 1)


def e4_quality(steps=1500, batch=64, lr=3e-3):
    # 학습 10만 시퀀스, 검증 2000. 표본이 많아 과적합이 아니라 구조(h_kv) 차이를 본다
    train, val = make_corpus(100_000, 0), make_corpus(2000, 1)  # 학습 표본을 넉넉히 둬서 과적합이 아니라 구조 차이를 본다
    out = {"entropy_floor": entropy_floor(), "steps": steps, "batch": batch, "lr": lr, "d": 32, "L": 2, "h": 4}
    for name, h_kv in (("MHA", 4), ("GQA", 2), ("MQA", 1)):
        losses = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            # seed마다 새 모델. torch.manual_seed가 가중치를 정한다
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
            # K·V 2개 * 층 2 * h_kv * dk 8(= d 32 / h 4)
            "cache_floats_per_token": 2 * 2 * h_kv * 8,  # K·V × 층 2 × h_kv × dk 8
        }
    return out


# ---------------------------------------------------------------- E5


def e5_bugs():
    p = model()
    # 올바른 캐시 생성이 기준
    ref_seq, ref_logits = kv.generate_with_cache(p, [0], 64, record_logits=True)

    # 버그 1: 새 토큰을 넣을 때 위치를 늘 0으로 준다(start를 빼먹음)
    cache = kv.new_cache(p, 65)
    seq, logs = [0], []
    logits = kv.forward(p, seq, 0, cache, last_only=True)
    for i in range(64):
        logs.append(logits)
        seq.append(int(np.argmax(logits)))
        if i < 63:
            # 버그: start를 0으로 준다. 위치 임베딩이 틀리고, 질의 위치가 0이라 마스크가 키 1.. 을 전부 가린다
            logits = kv.forward(p, seq[-1:], 0, cache, last_only=True)
    diff1 = [float(np.abs(a - b).max()) for a, b in zip(logs, ref_logits)]
    first_bad = next((i for i, (a, b) in enumerate(zip(seq, ref_seq)) if a != b), None)

    # 버그 2: 다른 프롬프트로 넘어가면서 캐시를 비우지 않는다
    cache = kv.new_cache(p, 200)
    # 먼저 다른 프롬프트로 캐시를 채운다(길이 3 + 20 - 1 = 22칸)
    kv.generate_with_cache(p, [5, 6, 7], 20, cache=cache)
    # 같은 캐시를 비우지 않고 다음 프롬프트 [0]을 넣는다: 새 k, v가 22번째 칸 뒤에 붙고 옛 칸이 어텐션에 섞인다
    stale_seq, stale_logits = kv.generate_with_cache(p, [0], 64, record_logits=True, cache=cache)
    # 첫 스텝 로짓이 이미 기준과 다르다
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
        # E1~E3, E5에 쓴 모델 설정
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
