"""8장 비교 실험. `uv run python llm/ch08_positional/experiments.py` 로 실행한다.

E0 실패 재현: 위치 정보 없는 self-attention의 순열 등변성(마스크 없음/있음), 위치를 넣으면 깨지는지
E1 손 계산 예: 2차원 사인 위치 벡터 (sin i, cos i), i = 0..3
E2 절대 위치 과제: 길이 32~64의 무작위 토큰열에서 "3번째 토큰"을 마지막 위치가 맞힌다. 5가지 방식 x 시드 3개
E2b 대조: 같은 과제를 길이 64로 고정하면 상대 위치 방식(rope, alibi)도 풀리는지
E3 상대 위치 과제와 외삽: 위치마다 "직전 토큰"을 맞힌다. 길이 64로 학습, 64/128/256에서 평가
E4 계산량: 방식별 추가 FLOP·파라미터(이론), 순전파 실측 시간

결과는 results/ch08.json 에 저장한다. 챕터 본문은 이 파일의 값을 그대로 인용한다.

입력: 없음(시드와 설정은 파일 상단 상수). 출력: results/ch08.json. 과제 설정(어휘 16, d = 64, 헤드 4, 학습 길이 64)은 아래 상수와 같다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import positional_np as pnp
from positional_torch import KINDS, PosAttentionModel

OUT = Path(__file__).parent / "results" / "ch08.json"
VOCAB, D, HEADS, N_TRAIN = 16, 64, 4, 64
STEPS, BATCH, LR = 1500, 32, 3e-3
TARGET_POS = 2  # 0부터 세므로 "3번째 토큰"


# E0: 순열 등변성 격차. 값이 0에 가까우면 f(Px) = P f(x), 즉 위치 정보가 없다.
def e0_equivariance():
    rng = np.random.default_rng(0)
    n, d = 8, 16
    x = rng.standard_normal((n, d))
    W = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3)]
    table = pnp.learned_pos(n, d, rng)
    perm = rng.permutation(n)
    out = {}
    for causal in (False, True):
        for kind in pnp.KINDS:
            f = lambda z: pnp.attention_with_pos(z, *W, kind=kind, pos_table=table, causal=causal)
            gap = float(np.abs(f(x[perm]) - f(x)[perm]).max())  # 0이면 "섞으면 같이 섞일 뿐"
            out[f"causal={causal},{kind}"] = gap
    # 배열 모양: 각 f의 출력은 (n, d_k) = (8, 16)
    return {"max_abs_gap_f(Px)_vs_Pf(x)": out, "n": n, "d": d, "perm": perm.tolist()}


def e1_toy():
    return pnp.first_screen_toy()


def batch_third(gen, batch, n):
    # x: (batch, n) 무작위 토큰 id. 정답은 항상 TARGET_POS = 2 위치의 토큰, (batch,)
    x = torch.randint(0, VOCAB, (batch, n), generator=gen)
    return x, x[:, TARGET_POS]


def batch_prev(gen, batch, n):
    x = torch.randint(0, VOCAB, (batch, n), generator=gen)
    # 정답 y: (batch, n-1). 위치 0은 직전 토큰이 없어 제외
    return x, x[:, :-1]  # 위치 i(1..n-1)의 정답은 x[i-1]


def loss_third(model, x, y):
    # model(x): (batch, n, vocab). 마지막 위치의 로짓만 정답과 비교
    return F.cross_entropy(model(x)[:, -1], y)


def loss_prev(model, x, y):
    # 위치 1..n-1의 로짓 (batch, n-1, vocab). y와 짝을 맞춘다(off-by-one 주의)
    logits = model(x)[:, 1:]
    return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))


def train(kind, seed, task, fixed_len=None):
    # 가중치 초기화 시드. 방식(kind)이 달라도 같은 시드를 써서 비교한다. 배치는 별도 generator(1000+seed)로 뽑는다
    torch.manual_seed(seed)
    model = PosAttentionModel(VOCAB, D, HEADS, kind, n_max=N_TRAIN)
    gen = torch.Generator().manual_seed(1000 + seed)
    len_gen = np.random.default_rng(2000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses, max_grad = [], 0.0
    t0 = time.perf_counter()
    for _ in range(STEPS):
        if task == "third":
            # 길이 32~64를 매 스텝 무작위로 뽑는다(상한 포함). fixed_len이 있으면 그 길이로 고정
            n = fixed_len or int(len_gen.integers(32, N_TRAIN + 1))
            x, y = batch_third(gen, BATCH, n)
            loss = loss_third(model, x, y)
        else:
            x, y = batch_prev(gen, BATCH, N_TRAIN)
            loss = loss_prev(model, x, y)
        opt.zero_grad()
        loss.backward()
        # 전체 기울기 L2 노름. 학습 중 최댓값을 기록해 폭주 여부를 본다
        g = float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
        max_grad = max(max_grad, g)
        opt.step()
        losses.append(loss.item())
    return model, {
        # 마지막 50스텝 평균 손실
        "final_train_loss": float(np.mean(losses[-50:])),
        "max_grad_norm": max_grad,
        "nan_loss": bool(np.isnan(losses).any()),
        "ms_per_step": 1000 * (time.perf_counter() - t0) / STEPS,
    }


@torch.no_grad()
def eval_third(model):
    # 평가용 시드 999 고정: 모든 모델이 같은 평가 데이터를 본다
    gen = torch.Generator().manual_seed(999)
    out = {}
    for n in (32, 48, 64):
        x, y = batch_third(gen, 1000, n)
        out[str(n)] = float((model(x)[:, -1].argmax(-1) == y).float().mean())
    return out


@torch.no_grad()
def eval_prev(model):
    gen = torch.Generator().manual_seed(999)
    out = {}
    for n in (64, 128, 256):
        x, y = batch_prev(gen, 200, n)
        # None을 JSON에 null로 남긴다
        if model.kind == "learned" and n > model.n_max:
            out[str(n)] = None  # 학습형 표에는 위치 64 이상의 벡터가 없어 평가할 수 없다
            continue
        logits, A = model(x, return_weights=True)
        logits = logits[:, 1:]
        # hit: (B, n-1). 열 j가 위치 j+1에 대응(위 슬라이스 때문에 한 칸 밀림)
        hit = (logits.argmax(-1) == y).float()  # (B, n-1), 열 j는 위치 j+1
        nll = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction="none").view_as(hit)
        # A: (B, heads, n, n). 질의 위치 1..n-1이 키 위치 0..n-2(직전 토큰)에 둔 가중치 -> (B, heads, n-1), 헤드 축 max
        prev_w = A[:, :, torch.arange(1, n), torch.arange(0, n - 1)].max(1).values  # 헤드 중 최대, 직전 칸 가중치
        # 열 번호 63부터가 위치 64 이상(열 j = 위치 j+1)
        new = slice(N_TRAIN - 1, None)  # 위치 64 이상(학습 때 본 적 없는 위치)
        out[str(n)] = {
            "acc_all": float(hit.mean()),
            "loss_all": float(nll.mean()),
            "acc_seen_positions": float(hit[:, : N_TRAIN - 1].mean()),
            "acc_new_positions": float(hit[:, new].mean()) if n > N_TRAIN else None,
            "loss_new_positions": float(nll[:, new].mean()) if n > N_TRAIN else None,
            "max_head_weight_on_prev_seen": float(prev_w[:, : N_TRAIN - 1].mean()),
            "max_head_weight_on_prev_new": float(prev_w[:, new].mean()) if n > N_TRAIN else None,
        }
    return out


def bag_baseline(task):
    """위치를 전혀 안 보는 규칙 하나: 지금까지 본 토큰 가운데 가장 많이 나온 것을 고른다(동률이면 번호가 작은 것)."""
    gen = torch.Generator().manual_seed(999)
    if task == "third":
        accs = {}
        for n in (32, 48, 64):
            x, y = batch_third(gen, 1000, n)
            # (B, n, V) -> 토큰별 등장 횟수 (B, V)
            counts = F.one_hot(x, VOCAB).sum(1)
            accs[str(n)] = float((counts.argmax(-1) == y).float().mean())
        return accs
    x, y = batch_prev(gen, 200, N_TRAIN)
    # cumsum으로 위치별 누적 횟수 (B, n, V), [:, 1:]로 위치 1.. 만 사용
    counts = F.one_hot(x, VOCAB).cumsum(1)[:, 1:]  # 위치 i까지(자기 자신 포함) 본 토큰 수
    return float((counts.argmax(-1) == y).float().mean())


def e2_third():
    out = {}
    for kind in KINDS:
        runs = []
        for s in range(3):
            model, log = train(kind, s, "third")
            runs.append({**log, "test_acc_by_len": eval_third(model)})
        out[kind] = runs
    # chance = 1/어휘 크기. bag_baseline은 위치 없이 빈도만 쓰는 규칙의 정확도
    return {"chance": 1 / VOCAB, "bag_baseline_acc_by_len": bag_baseline("third"), "train_lengths": [32, N_TRAIN], "runs": out}


def e2b_third_fixed_len():
    """대조 실험: 길이를 64로 고정하면 3번째 토큰까지의 거리가 늘 61이라 상대 위치만으로도 찾을 수 있다."""
    gen_eval = lambda: torch.Generator().manual_seed(999)
    out = {}
    # 상대 위치 방식 두 가지와 위치 없음만 대조
    for kind in ("none", "rope", "alibi"):
        runs = []
        for s in range(3):
            model, log = train(kind, s, "third", fixed_len=N_TRAIN)
            with torch.no_grad():
                x, y = batch_third(gen_eval(), 1000, N_TRAIN)
                acc = float((model(x)[:, -1].argmax(-1) == y).float().mean())
            runs.append({**log, "test_acc_len64": acc})
        out[kind] = runs
    return {"train_length": N_TRAIN, "runs": out}


def e3_prev():
    out = {}
    for kind in KINDS:
        runs = []
        for s in range(3):
            model, log = train(kind, s, "prev")
            runs.append({**log, "eval": eval_prev(model)})
        out[kind] = runs
    return {"chance": 1 / VOCAB, "bag_baseline_acc_len64": bag_baseline("prev"), "train_length": N_TRAIN, "runs": out}


def _median_ms(fn, repeats=15):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


@torch.no_grad()
def e4_cost():
    rows = []
    for n in (64, 256, 1024):
        x = torch.randint(0, VOCAB, (8, n), generator=torch.Generator().manual_seed(0))
        for kind in KINDS:
            # n_max = n으로 두어 학습형 표 크기를 해당 길이에 맞춘다
            flops, params = pnp.extra_cost(kind, n, D, HEADS, n_max=n)
            row = {"n": n, "kind": kind, "extra_flops": flops, "extra_params_if_n_max=n": params}
            torch.manual_seed(0)
            m = PosAttentionModel(VOCAB, D, HEADS, kind, n_max=n)
            m(x)
            # 위치 방식에 따른 전체 파라미터 수 차이(학습형만 n_max*d 더 큼)
            row["model_params"] = sum(p.numel() for p in m.parameters())
            row["forward_ms_batch8"] = _median_ms(lambda: m(x))
            rows.append(row)
    # 비교 기준: 위치와 무관한 한 층의 연산(7장 attention_flops와 같은 셈, 헤드를 나눠도 합은 같다)
    # 6nD^2(Q,K,V 투영) + 4n^2 D(점수 2n^2 D + 혼합 2n^2 D). 이 모델은 출력 투영 W_O가 없어 7장 attention_flops와 같은 식
    base = {str(n): 6 * n * D * D + 4 * n * n * D for n in (64, 256, 1024)}
    return {"rows": rows, "attention_layer_flops": base, "note": "시간 열은 기기마다 다르다. 단일 스레드."}


def main():
    # 단일 스레드로 고정해 시간 비교가 가능하게 한다
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    results = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "torch_threads": 1,
        },
        "config": {"vocab": VOCAB, "d": D, "heads": HEADS, "train_len": N_TRAIN, "steps": STEPS, "batch": BATCH, "lr": LR,
                   "alibi_slopes": pnp.alibi_slopes(HEADS).tolist()},
        "E0": e0_equivariance(),
        "E1": e1_toy(),
        "E2": e2_third(),
        "E2b": e2b_third_fixed_len(),
        "E3": e3_prev(),
        "E4": e4_cost(),
    }
    results["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
