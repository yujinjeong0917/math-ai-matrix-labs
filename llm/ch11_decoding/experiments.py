"""11장 비교 실험. `uv run python llm/ch11_decoding/experiments.py` 로 실행한다.

정답 언어: 앞 두 토큰이 다음 토큰의 분포를 정하는 합성 언어(어휘 48, decoding_np.SecondOrderSource).
정답 확률을 정확히 알기 때문에 "사람이 쓴 글" 자리에 이 언어의 진짜 표본을 두고 비교할 수 있다.
모델: 9장 Pre-LN 소형 GPT(2층, d=64, 헤드 4, 문맥 64)를 이 언어의 표본으로 학습한다.
10장 체크포인트가 아직 없어서 이 장에서 직접 학습한다(설계도와 다른 점).

E0 학습: 정답 언어의 엔트로피(이론 최저 손실) vs 모델의 검증 손실
E1 실패 재현 + E2 비교: 같은 프롬프트 20개, 각 200토큰.
   {탐욕, 빔 4, T 0.7/1.0, top-k 10/40, top-p 0.9/0.95} 와 정답 언어의 진짜 이어 쓰기(기준).
   지표: 4-gram 중복 비율(rep4), distinct-2, 모델 하 토큰당 평균 로그확률(T=1, 자르지 않은 원래 분포),
   정답 언어에서 나올 수 없는 토큰의 비율(불가능 토큰), 생성 속도(토큰/초). 표본추출은 시드 5개.
E1b 완벽한 모델: 정답 언어의 분포 자체로 탐욕·빔 4·순수 표본추출을 하면
E1c 빔 크기 1, 2, 4, 8(완벽한 모델): 누적 로그확률과 반복 비율
E3 탐욕이 빠지는 고리: 생성문 끝 60토큰이 정확히 같은 주기로 도는지
E4 nucleus 크기가 문맥마다 바뀌는지: top-p 0.9 집합 크기 vs 정답 언어의 허용 토큰 수
E5 반복이 반복을 부르나(Holtzman 등 Figure 4의 작은 재현): 같은 구절을 k번 반복한 뒤 다음 반복의 확률

결과는 results/ch11.json 에 저장하고, 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import decoding_np as dn
from decoding_torch import TinyLM, numpy_model

OUT = Path(__file__).parent / "results" / "ch11.json"
V, N_CTX, D, L, H = 48, 64, 64, 2, 4
N_TRAIN_SEQ, STEPS, BATCH, LR = 10000, 3000, 64, 3e-3
N_PROMPTS, PROMPT_LEN, GEN_LEN, BEAM = 20, 16, 200, 4
SEEDS = range(5)
SOURCE = dn.SecondOrderSource(V=V, seed=0)

METHODS = {
    "greedy": None,
    "beam4": None,
    "T=0.7": dict(T=0.7),
    "T=1.0": dict(T=1.0),
    "top-k=10": dict(k=10),
    "top-k=40": dict(k=40),
    "top-p=0.9": dict(p=0.9),
    "top-p=0.95": dict(p=0.95),
}


# ---------------------------------------------------------------- E0 학습


def train_model():
    rng = np.random.default_rng(100)
    train = torch.tensor(SOURCE.sample(N_TRAIN_SEQ, N_CTX + 1, rng))
    val = torch.tensor(SOURCE.sample(500, N_CTX + 1, np.random.default_rng(101)))
    torch.manual_seed(0)
    model = TinyLM(V, N_CTX, D, L, H)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)

    def loss_on(X):
        logits = model(X[:, :-1])[:, 1:]  # 첫 두 토큰은 무작위 시작이라, 세 번째 토큰부터 예측을 잰다
        return F.cross_entropy(logits.reshape(-1, V), X[:, 2:].reshape(-1))

    t0 = time.perf_counter()
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / 100) * 0.5 * (1 + np.cos(np.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, N_TRAIN_SEQ, (BATCH,), generator=gen)
        loss = loss_on(train[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    secs = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        tr, va = float(loss_on(train[:500])), float(loss_on(val))
    return model, {
        "source_entropy_rate_nat": SOURCE.entropy_rate(),
        "train_loss_nat": tr,
        "val_loss_nat": va,
        "train_tokens": N_TRAIN_SEQ * N_CTX,
        "steps": STEPS,
        "train_seconds": secs,
        "params": sum(p.numel() for p in model.parameters()),
        "config": dict(V=V, n_ctx=N_CTX, d=D, layers=L, heads=H, batch=BATCH, lr=LR),
    }


# ---------------------------------------------------------------- E1/E2 생성 비교


def score(model_fn, ids):
    cont = ids[:, PROMPT_LEN:]
    reps = [dn.repetition_stats(row) for row in cont]
    lp = dn.sequence_logprob(model_fn, ids, PROMPT_LEN)
    true_p = SOURCE.token_probs(ids, PROMPT_LEN)
    return {
        "rep4": float(np.mean([r["rep4"] for r in reps])),
        "distinct2": float(np.mean([r["distinct2"] for r in reps])),
        "model_logprob_per_token": float(lp.mean()),
        "impossible_token_rate": float((true_p == 0).mean()),
        "true_logprob_per_token_possible_only": float(np.log(true_p[true_p > 0]).mean()),
    }


def mean_sd(runs):
    keys = runs[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs])), "runs": [r[k] for r in runs]} for k in keys}


def e12_generate(model_fn, prompts):
    out = {}
    # 기준: 정답 언어가 같은 프롬프트에서 직접 이어 쓴 글(시드 5개)
    out["reference"] = mean_sd([
        score(model_fn, SOURCE.sample(N_PROMPTS, PROMPT_LEN + GEN_LEN, np.random.default_rng(500 + s), start=prompts))
        for s in SEEDS
    ])
    samples = {}
    for name, cfg in METHODS.items():
        t0 = time.perf_counter()
        if name == "greedy":
            ids = dn.greedy(model_fn, prompts, GEN_LEN)
            secs = time.perf_counter() - t0
            r = score(model_fn, ids)
            r["tokens_per_sec"] = N_PROMPTS * GEN_LEN / secs
            out[name] = mean_sd([r])
        elif name == "beam4":
            seqs = [dn.beam_search(model_fn, pr, GEN_LEN, BEAM)[0] for pr in prompts]
            secs = time.perf_counter() - t0
            ids = np.stack(seqs)
            r = score(model_fn, ids)
            r["tokens_per_sec"] = N_PROMPTS * GEN_LEN / secs
            out[name] = mean_sd([r])
        else:
            runs = []
            for s in SEEDS:
                t0 = time.perf_counter()
                ids = dn.sample(model_fn, prompts, GEN_LEN, np.random.default_rng(s), **cfg)
                secs = time.perf_counter() - t0
                r = score(model_fn, ids)
                r["tokens_per_sec"] = N_PROMPTS * GEN_LEN / secs
                runs.append(r)
                if s == 0:
                    samples[name] = ids
            out[name] = mean_sd(runs)
            continue
        samples[name] = ids
    return out, samples


# ---------------------------------------------------------------- E3 탐욕의 고리


def period_at_end(row, max_period=100, tail=60):
    """끝의 tail개 토큰이 주기 P로 반복되는 가장 작은 P. 없으면 None."""
    row = list(row)
    end = row[-tail:]
    for P in range(1, max_period + 1):
        if all(end[i] == end[i - P] for i in range(P, len(end))) and len(row) >= tail + P:
            if row[-tail - P : -tail] == end[:P]:
                return P
    return None


def source_logits(ids):
    """정답 언어를 그대로 '완벽한 모델'로 쓴다. 나올 수 없는 토큰은 -inf."""
    ids = np.atleast_2d(ids)
    p = SOURCE.P[ids[:, -2], ids[:, -1]]
    with np.errstate(divide="ignore"):
        return np.log(p)


def e1b_perfect_model(model_fn, prompts):
    """모델이 정답 분포를 완벽히 배웠다고 해도 최대화 디코딩은 반복하나."""
    g = dn.greedy(source_logits, prompts, GEN_LEN)
    b = np.stack([dn.beam_search(source_logits, pr, GEN_LEN, BEAM)[0] for pr in prompts])
    s = np.stack([dn.sample(source_logits, prompts, GEN_LEN, np.random.default_rng(s_)) for s_ in SEEDS])
    out = {}
    for name, ids in (("perfect_greedy", g), ("perfect_beam4", b)):
        r = score(model_fn, ids)
        r["periods"] = [period_at_end(row[PROMPT_LEN:]) for row in ids]
        out[name] = r
    out["perfect_sampling_T1"] = mean_sd([score(model_fn, ids) for ids in s])
    return out


def e1c_beam_sweep(prompts):
    """완벽한 모델에서 빔 크기 1, 2, 4, 8: 누적 로그확률과 반복 비율(연습 2의 정답 기준)."""
    out = {}
    for b in (1, 2, 4, 8):
        res = [dn.beam_search(source_logits, pr, GEN_LEN, b) for pr in prompts]
        out[str(b)] = {
            "mean_total_logprob": float(np.mean([r[1] for r in res])),
            "rep4": float(np.mean([dn.repetition_stats(r[0][PROMPT_LEN:])["rep4"] for r in res])),
            "per_prompt_total_logprob": [r[1] for r in res],
        }
    lp = {b: np.array(out[b]["per_prompt_total_logprob"]) for b in out}
    out["prompts_where_wider_beam_scored_lower"] = {
        f"{a}->{b}": int((lp[b] < lp[a] - 1e-9).sum()) for a, b in (("1", "2"), ("2", "4"), ("4", "8"))
    }
    return out


def e3_loops(samples):
    out = {}
    for name in ("greedy", "beam4", "T=0.7", "top-p=0.9"):
        periods = [period_at_end(r[PROMPT_LEN:]) for r in samples[name]]
        stuck = [p for p in periods if p is not None]
        out[name] = {"stuck_prompts": len(stuck), "of": N_PROMPTS, "periods": periods}
    return out


# ---------------------------------------------------------------- E4 nucleus 크기


def e4_nucleus(model_fn, prompts):
    ids = SOURCE.sample(N_PROMPTS, PROMPT_LEN + GEN_LEN, np.random.default_rng(900), start=prompts)
    sizes, support, k40_impossible_mass, nuc_impossible_mass, full_impossible_mass = [], [], [], [], []
    for t in range(PROMPT_LEN, PROMPT_LEN + GEN_LEN, 4):
        logits = model_fn(ids[:, :t])
        probs = dn.softmax(logits)
        nuc = np.isfinite(dn.top_p_filter(logits, 0.9))
        k40 = np.isfinite(dn.top_k_filter(logits, 40))
        allowed = SOURCE.P[ids[:, t - 2], ids[:, t - 1]] > 0
        sizes += nuc.sum(-1).tolist()
        support += allowed.sum(-1).tolist()
        for mask, acc in ((nuc, nuc_impossible_mass), (k40, k40_impossible_mass), (np.ones_like(nuc), full_impossible_mass)):
            q = np.where(mask, probs, 0)
            q = q / q.sum(-1, keepdims=True)
            acc += (q * ~allowed).sum(-1).tolist()  # 다듬은 분포가 '나올 수 없는 토큰'에 준 확률
    sizes, support = np.array(sizes), np.array(support)
    return {
        "nucleus_size": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()), "mean": float(sizes.mean())},
        "true_support": {"min": int(support.min()), "median": float(np.median(support)), "max": int(support.max())},
        "corr_nucleus_vs_support": float(np.corrcoef(sizes, support)[0, 1]),
        "impossible_mass_mean": {
            "full": float(np.mean(full_impossible_mass)),
            "top-k=40": float(np.mean(k40_impossible_mass)),
            "top-p=0.9": float(np.mean(nuc_impossible_mass)),
        },
        "n_contexts": int(len(sizes)),
    }


# ---------------------------------------------------------------- E5 반복이 반복을 부르나


def e5_feedback(model_fn):
    rng = np.random.default_rng(700)
    phrases = SOURCE.sample(50, 6, rng)[:, 2:]  # 정답 언어에서 나온 4토큰 구절 50개
    prefix = SOURCE.sample(50, 8, np.random.default_rng(701))
    by_k = {}
    for k in range(1, 9):
        ids = np.concatenate([prefix] + [phrases] * k + [phrases], axis=1)
        start = ids.shape[1] - 4
        lp = dn.sequence_logprob(model_fn, ids, start)  # 마지막(k+1번째) 반복 4토큰의 확률
        by_k[str(k)] = float(np.exp(lp).mean())
    return {"mean_prob_next_repetition_by_k": by_k, "n_phrases": 50, "phrase_len": 4}


def examples(samples):
    return {name: samples[name][0].tolist() for name in ("greedy", "beam4", "T=1.0", "top-p=0.9")}


def main():
    torch.set_num_threads(1)
    model, e0 = train_model()
    model_fn = numpy_model(model)
    prompts = SOURCE.sample(N_PROMPTS, PROMPT_LEN, np.random.default_rng(300))
    e12, samples = e12_generate(model_fn, prompts)
    results = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "torch_threads": 1,
        },
        "setup": dict(
            n_prompts=N_PROMPTS, prompt_len=PROMPT_LEN, gen_len=GEN_LEN, beam=BEAM, sampling_seeds=len(SEEDS),
            # 모델에 넣은 문맥 행 수(토큰 하나를 만들 때마다 문맥 전체를 다시 넣는다. KV cache 없음)
            forward_rows={"greedy_or_sampling": N_PROMPTS * GEN_LEN, "beam4": N_PROMPTS * (1 + BEAM * (GEN_LEN - 1))},
            speed_note="탐욕·표본추출은 프롬프트 20개를 한 번에(배치 20), 빔은 프롬프트마다 따로(배치 4) 돌렸다.",
        ),
        "E0": e0,
        "E12": e12,
        "E1b": e1b_perfect_model(model_fn, prompts),
        "E1c": e1c_beam_sweep(prompts),
        "E3": e3_loops(samples),
        "E4": e4_nucleus(model_fn, prompts),
        "E5": e5_feedback(model_fn),
        "example_first_prompt": examples(samples),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    brief = {k: {m: round(v["mean"], 4) for m, v in r.items()} for k, r in e12.items()}
    print(json.dumps({"E0": e0, "E12": brief, "E1b": results["E1b"], "E3": results["E3"], "E4": results["E4"], "E5": results["E5"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
