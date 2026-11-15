"""6장 비교 실험. `uv run python llm/ch06_attention_seq2seq/experiments.py` 로 실행한다.

E0 손 계산 예: 인코더 상태 3개, 곱셈형 점수 -> softmax 비율 -> 문맥 벡터 (첫 화면 숫자)
E1 경로와 기울기: 학습 전, 첫 출력의 점수(logits)를 첫 입력 임베딩으로 미분한 노름. 고정 문맥 vs 곱셈형 어텐션
E2 학습 비교: 복사 과제에서 고정 문맥 / 덧셈형 / dot / general, 역순 과제에서 고정 문맥 / dot. 시드 3개
E3 정렬: 학습된 어텐션 모델의 alpha가 대각선(복사) / 반대 대각선(역순)에 몰리는지
E4 계산량: 디코딩 한 걸음의 점수 계산 수와 문장 전체의 점수 수, NumPy 실측 디코딩 시간

결과는 results/ch06.json 에 저장한다. 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import seq2seq_np as snp
from seq2seq_torch import Seq2Seq

OUT = Path(__file__).parent / "results" / "ch06.json"
VOCAB = 20
# 학습 길이를 5~40에서 섞어 뽑고, 평가는 같은 범위의 4개 길이만 쓴다(범위 밖 일반화는 재지 않는다).
TRAIN_LEN = (5, 40)  # 배치마다 길이를 이 범위에서 고른다. 평가 길이는 모두 이 범위 안
EVAL_LENS = (5, 10, 20, 40)
STEPS, BATCH, LR, CLIP = 2500, 64, 3e-3, 1.0
SEEDS = (0, 1, 2)


def e0_hand_example():
    """첫 화면 손 계산: 점수 [2, 1, 3] -> exp 합 -> alpha -> 문맥. 고정 문맥(마지막 상태 h_L)과 비교한다."""
    # 3개의 인코더 상태 (L=3, d=2). 점수 s^T h = [2, 1, 3].
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 단어 세 개를 읽은 뒤의 메모(인코더 상태)
    s = np.array([2.0, 1.0])  # 지금 디코더가 들고 있는 상태
    scores = snp.dot_score(s, H)
    c, alpha = snp.attend(scores, H)
    return {
        "H": H.tolist(), "s": s.tolist(), "scores": scores.tolist(),
        "exp_scores": np.exp(scores).tolist(), "exp_sum": float(np.exp(scores).sum()),
        "alpha": alpha.tolist(), "context": c.tolist(), "fixed_context_h_L": H[-1].tolist(),
    }


def make_batch(gen, batch, L, task):
    """복사 또는 역순 과제 배치. x: (B, L) 입력, y: 정답, tgt_in: BOS + y[:-1]. 토큰 번호 VOCAB(=20)이 BOS."""
    x = torch.randint(0, VOCAB, (batch, L), generator=gen)
    y = x if task == "copy" else x.flip(1)
    tgt_in = torch.cat([torch.full((batch, 1), VOCAB), y[:, :-1]], dim=1)  # 디코더 입력: BOS, y_1, ..., y_{L-1}
    return x, y, tgt_in


def e1_gradient_path():
    """학습 전 가중치를 하나 고정하고 입력만 20번 새로 뽑아, ||d logits_1 / d emb(x_1)||_2 를 평균 낸다.

    고정 문맥: x_1 -> h_1 -> h_2 -> ... -> h_L -> 디코더. 인코더 L걸음을 지난다.
    어텐션:   x_1 -> h_1 -> c_1 -> 디코더. 인코더 한 걸음 뒤 바로 문맥으로 간다(h_L 경로도 함께 있다).
    """
    out = {}
    for score in ("fixed", "dot"):
        torch.manual_seed(0)
        m = Seq2Seq(VOCAB, score=score).double()
        g = torch.Generator().manual_seed(7)
        res = {}
        for L in EVAL_LENS + (80,):
            norms = []
            for _ in range(20):
                x = torch.randint(0, VOCAB, (1, L), generator=g)
                xe = m.emb(x).detach()

                # 첫 입력 임베딩만 변수로 두고 나머지 위치는 고정. 첫 출력 로짓 (V,)를 반환하는 함수라 야코비안이 (V, d).
                def f(first):
                    z = torch.cat([first[None, None], xe[:, 1:]], dim=1)
                    H, hL = m.encode_embedded(z)
                    logits, _ = m.decode_teacher_forced(H, hL, torch.full((1, 1), VOCAB))
                    return logits[0, 0]

                # J: (VOCAB, d). 로짓 1개 묶음이 첫 입력 임베딩에 얼마나 민감한지.
                J = torch.autograd.functional.jacobian(f, xe[0, 0])
                # 스펙트럼 노름(최대 특이값). 입력만 바꿔 20번 평균.
                norms.append(float(torch.linalg.matrix_norm(J, ord=2)))
            # times_L = 노름 x L. 거리에 따라 줄어드는 정도를 L과 함께 보려는 보조 열(해석은 본문 확인 필요).
            res[str(L)] = {"mean": float(np.mean(norms)), "std": float(np.std(norms)), "times_L": float(np.mean(norms) * L)}
        out[score] = res
    return out


def train(score, task, seed, steps=STEPS):
    """점수 방식 하나, 과제 하나, 시드 하나로 학습하고 길이별 정확도와 정렬을 잰다."""
    torch.manual_seed(seed)
    m = Seq2Seq(VOCAB, score=score)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(100 + seed)
    losses, max_grad, first_hit, curve = [], 0.0, None, {}
    t0 = time.perf_counter()
    for step in range(steps):
        # 배치마다 길이 L을 새로 뽑는다(한 배치 안의 길이는 같다).
        L = int(torch.randint(TRAIN_LEN[0], TRAIN_LEN[1] + 1, (1,), generator=gen))
        x, y, tgt_in = make_batch(gen, BATCH, L, task)
        logits, _ = m(x, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        # clip_grad_norm_은 자르기 전 전체 노름을 돌려준다. 그 값을 기록하고 같은 호출로 CLIP(1.0)에 맞춰 자른다.
        g = float(torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP))  # 자르기 전 노름을 돌려준다
        max_grad = max(max_grad, g)
        opt.step()
        losses.append(loss.item())
        # 최근 50스텝 평균 손실이 0.1 미만이 된 첫 스텝(수렴 속도 지표).
        if first_hit is None and step >= 50 and np.mean(losses[-50:]) < 0.1:
            first_hit = step
        if (step + 1) % 500 == 0:
            curve[str(step + 1)] = float(np.mean(losses[-50:]))
    sec = time.perf_counter() - t0

    test_gen = torch.Generator().manual_seed(999)
    by_len = {}
    for L in EVAL_LENS:
        x, y, _ = make_batch(test_gen, 500, L, task)
        # 평가는 teacher forcing 없이 모델 자신의 예측을 되먹이는 탐욕 디코딩. A: (B, L, L) = (출력 위치, 입력 위치).
        pred, A = m.greedy(x, L)
        hit = (pred == y).float()
        row = {"token_acc": float(hit.mean()), "exact_match": float(hit.all(1).float().mean())}
        # 가장 긴 길이에서만 출력 위치 구간별 정확도를 본다(뒤로 갈수록 떨어지는지).
        if L == 40:  # 출력 위치별 정확도: 10칸씩 묶는다
            pos = hit.mean(0)
            row["token_acc_by_output_position"] = {f"{a+1}-{a+10}": float(pos[a : a + 10].mean()) for a in range(0, 40, 10)}
        if A is not None:
            # 정렬의 정답: 복사는 출력 j <-> 입력 j, 역순은 출력 j <-> 입력 L-1-j.
            target = torch.arange(L) if task == "copy" else torch.arange(L).flip(0)
            row["alpha_argmax_on_alignment"] = float((A.argmax(-1) == target).float().mean())
            # 정답 정렬 위치에 실린 alpha 질량의 평균(argmax 일치율보다 부드러운 지표).
            row["alpha_mass_on_alignment"] = float(A[:, torch.arange(L), target].mean())
            # argmax가 정답 위치에서 -1, 0, +1칸 옆인 비율을 따로 센다.
            offset = A.argmax(-1) - target  # 정답 위치에서 몇 칸 옆을 가장 크게 봤나
            row["alpha_argmax_offset_share"] = {str(k): float((offset == k).float().mean()) for k in (-1, 0, 1)}
        by_len[str(L)] = row
    return {
        "by_length": by_len,
        "final_loss": float(np.mean(losses[-50:])),
        "loss_curve_every_500": curve,
        "steps_to_loss_below_0.1": first_hit,
        "max_grad_norm_before_clip": max_grad,
        "nan_loss": bool(np.isnan(losses).any()),
        "ms_per_step": 1000 * sec / steps,
        "params": int(sum(p.numel() for p in m.parameters())),
    }, m


def e2_e3_training():
    """복사 4종, 역순 2종 설정을 시드 3개씩 학습한다. 정렬 그림용 alpha 한 개를 dot, 시드 0에서 저장한다."""
    runs, align_example = {}, {}
    plan = [("copy", s) for s in ("fixed", "additive", "dot", "general")] + [("reverse", s) for s in ("fixed", "dot")]
    for task, score in plan:
        key = f"{task},{score}"
        runs[key] = []
        for seed in SEEDS:
            r, m = train(score, task, seed)
            runs[key].append(r)
            print(key, seed, {L: round(v["exact_match"], 3) for L, v in r["by_length"].items()}, f"{r['ms_per_step']:.1f}ms", flush=True)
            # 시연용 한 개만 저장한다(통계 근거가 아님).
            if score == "dot" and seed == 0:  # 정렬 그림용: 길이 8 입력 하나의 alpha 행렬
                g = torch.Generator().manual_seed(5)
                x, y, _ = make_batch(g, 1, 8, task)
                pred, A = m.greedy(x, 8)
                align_example[task] = {"src": x[0].tolist(), "pred": pred[0].tolist(), "alpha": [[round(float(a), 3) for a in row] for row in A[0]]}
    summary = {}
    for key, rs in runs.items():
        summary[key] = {
            L: {
                "token_acc_mean": float(np.mean([r["by_length"][L]["token_acc"] for r in rs])),
                "exact_match_mean": float(np.mean([r["by_length"][L]["exact_match"] for r in rs])),
                "exact_match_by_seed": [r["by_length"][L]["exact_match"] for r in rs],
            }
            for L in map(str, EVAL_LENS)
        }
    return {"runs": runs, "summary": summary, "alignment_example_len8_seed0": align_example}


def e4_compute():
    """디코딩 한 걸음의 점수 비용과 문장 전체 비용. 실측은 NumPy 한 문장(배치 없음), 출력 길이 = 입력 길이."""
    d = 64
    rng = np.random.default_rng(0)
    gru_e = (rng.standard_normal((3 * d, d)) * 0.1, rng.standard_normal((3 * d, d)) * 0.1, np.zeros(3 * d), np.zeros(3 * d))
    gru_d = (rng.standard_normal((3 * d, d)) * 0.1, rng.standard_normal((3 * d, d)) * 0.1, np.zeros(3 * d), np.zeros(3 * d))
    Wc, Wo = rng.standard_normal((d, 2 * d)) * 0.1, rng.standard_normal((VOCAB, d)) * 0.1
    Ws, Wh, v = rng.standard_normal((d, d)) * 0.1, rng.standard_normal((d, d)) * 0.1, rng.standard_normal(d) * 0.1
    E = rng.standard_normal((VOCAB + 1, d)) * 0.1

    def decode(H, L, score, params):
        s, y = H[-1], VOCAB
        for _ in range(L):
            logits, s, _ = snp.seq2seq_attention_step(E[y], s, H, gru_d, Wc, Wo, score=score, score_params=params)
            y = int(np.argmax(logits))

    rows = []
    for L in (10, 40, 160, 640):
        xs = E[rng.integers(0, VOCAB, L)]
        H = snp.encode(xs, gru_e)
        # 점수 개수: 출력 L걸음 x 걸음당 L개 = L^2.
        # gru_flops_per_step: 입력 행렬곱과 상태 행렬곱(각 3d x d)을 곱셈+덧셈 2회로 센 값. 활성화 함수는 제외.
        row = {"L": L, "d": d, "scores_per_sentence": L * L, "gru_flops_per_step": 2 * 3 * d * d * 2}
        # additive의 마지막 인자 H @ Wh.T는 디코더 걸음마다 같아 미리 계산해 넘긴다(additive_score의 HWh).
        for score, params in (("fixed", ()), ("dot", ()), ("additive", (Ws, Wh, v, H @ Wh.T))):  # Wh h_i 는 미리 계산
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                decode(H, L, score, params)
                times.append(time.perf_counter() - t0)
            row[f"{score}_decode_ms"] = 1000 * float(np.median(times))
            if score != "fixed":
                row[f"{score}_score_flops_per_step"] = snp.score_flops(L, d, kind=score)
        rows.append(row)
    return {"rows": rows, "note": "덧셈형은 Wh h_i를 문장마다 한 번 미리 계산(2 L d^2, 표에는 넣지 않음). gru_flops_per_step는 GRU 셀의 두 행렬곱(입력·상태, 각 3d x d)만 센 값. 시간 열은 기기마다 다르다."}


def main():
    """결과를 모아 results/ch06.json에 쓴다(E2, E3는 학습이라 가장 오래 걸린다)."""
    torch.set_num_threads(1)
    results = {
        "env": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "machine": platform.machine(), "torch_threads": 1,
        },
        "setup": {
            "vocab": VOCAB, "train_len_range": TRAIN_LEN, "eval_lens": EVAL_LENS, "hidden": 64, "emb": 32,
            "steps": STEPS, "batch": BATCH, "lr": LR, "grad_clip_norm": CLIP, "seeds": SEEDS,
            "encoder": "단방향 GRU", "decoder_flow": "s_t = GRU(y_{t-1}, s_{t-1}) 뒤 s_t로 점수(Luong 흐름)",
        },
        "E0": e0_hand_example(),
        "E1": e1_gradient_path(),
        "E4": e4_compute(),
    }
    # 오래 걸리는 학습은 가벼운 E0, E1, E4를 끝낸 뒤에 돈다.
    results.update({"E2_E3": e2_e3_training()})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in ("E0", "E1", "E4")}, indent=1, ensure_ascii=False))
    print(json.dumps(results["E2_E3"]["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
