"""넘침·사라짐 재현, loss scaling, 동적 스케일러, fp32 주 가중치(master weights) 흉내.

가정: 기울기는 실제 프레임워크처럼 '계산한 뒤 fp16으로 저장'된다고 본다. 곱셈 중간값은
fp64로 계산하고 저장할 때 한 번만 반올림한다(실제 GPU 커널의 중간 누적 정밀도는 커널마다 다르다).
"""

import math
import random

from ft_ch04_formats import bf16, f16, f32

# PyTorch torch.amp.GradScaler 기본값(GitHub main 소스의 docstring, 2026-10-11 확인)
GRADSCALER_DEFAULTS = {"init_scale": 2.0 ** 16, "growth_factor": 2.0, "backoff_factor": 0.5,
                       "growth_interval": 2000}


# ---------- 장난감 모델: 손실 ½(wx − y)² ----------

def toy_grad(w, x, y, scale=1.0):
    """scale × ½(wx − y)² 를 w로 미분하면 scale × (wx − y) × x (연쇄법칙으로 scale이 그대로 곱해진다)."""
    return scale * (w * x - y) * x


def grad_through_fp16(w, x, y, scale):
    """기울기를 fp16으로 저장했다가 fp32에서 scale로 나눠 되돌린다."""
    stored = f16(toy_grad(w, x, y, scale))
    if math.isinf(stored) or math.isnan(stored):
        return {"scale": scale, "stored_fp16": stored, "unscaled_fp32": None, "overflow": True}
    back = f32(stored / scale)
    true = toy_grad(w, x, y)
    return {"scale": scale, "stored_fp16": stored, "unscaled_fp32": back, "overflow": False,
            "true": true, "rel_err": None if back == 0 else abs(back - true) / abs(true),
            "lost": back == 0}


# ---------- 기울기 묶음: 크기가 여러 자릿수에 걸친 가상 기울기 ----------

def synthetic_grads(n=10_000, lo=-10, hi=-1, seed=4):
    """10^lo ~ 10^hi 사이에서 로그 균등하게 뽑은 가상 기울기 크기(부호는 무작위). 실제 모델의 분포가 아니다."""
    rng = random.Random(seed)
    return [rng.choice((-1, 1)) * 10 ** rng.uniform(lo, hi) for _ in range(n)]


def census(grads, scale):
    """scale을 곱해 fp16으로 저장했을 때 0이 된 것, 비정규수, 정규수, 넘친 것의 개수."""
    out = {"zero": 0, "subnormal": 0, "normal": 0, "overflow": 0}
    min_normal = 2.0 ** -14
    for g in grads:
        s = f16(g * scale)
        if math.isinf(s):
            out["overflow"] += 1
        elif s == 0:
            out["zero"] += 1
        elif abs(s) < min_normal:
            out["subnormal"] += 1
        else:
            out["normal"] += 1
    return out


# ---------- 동적 loss scaling (GradScaler 흉내) ----------

class DynamicScaler:
    """torch.amp.GradScaler의 규칙만 옮긴 흉내.

    - 기울기 × scale을 fp16으로 저장한다.
    - inf/NaN이 하나라도 있으면 그 스텝의 업데이트를 건너뛰고 scale × backoff_factor.
    - 건너뛰지 않은 스텝이 growth_interval번 이어지면 scale × growth_factor.
    """

    def __init__(self, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.good = 0

    def step(self, grads):
        scale_used = self.scale
        stored = [f16(g * scale_used) for g in grads]
        found_inf = any(math.isinf(s) or math.isnan(s) for s in stored)
        if found_inf:
            self.scale *= self.backoff_factor
            self.good = 0
            unscaled = None
        else:
            unscaled = [f32(s / scale_used) for s in stored]
            self.good += 1
            if self.good == self.growth_interval:
                self.scale *= self.growth_factor
                self.good = 0
        return {"scale_used": scale_used, "max_scaled": max(abs(g) * scale_used for g in grads),
                "skipped": found_inf, "next_scale": self.scale, "unscaled": unscaled}


def scaler_trace(grad_steps, **kw):
    sc = DynamicScaler(**kw)
    return [sc.step(g) for g in grad_steps]


# ---------- 작은 업데이트가 사라지는 문제와 fp32 주 가중치 ----------

def accumulate(update, steps, fmt, w0=1.0):
    """w ← w + update 를 steps번. 가중치를 그 형식에 저장하면 간격의 절반보다 작은 업데이트는 사라진다."""
    rnd = {"fp32": f32, "fp16": f16, "bf16": bf16}[fmt]
    w = rnd(w0)
    u = rnd(update)
    for _ in range(steps):
        w = rnd(w + u)
    return w


def master_weight_run(update, steps, w0=1.0, low="bf16"):
    """fp32 주 가중치에 업데이트를 쌓고, 계산용 사본만 낮은 형식으로 반올림한다."""
    rnd = {"fp16": f16, "bf16": bf16}[low]
    master = f32(w0)
    for _ in range(steps):
        master = f32(master + f32(update))
    return {"master_fp32": master, "copy_low": rnd(master)}


# ---------- 넘침: 곱셈과 exp ----------

def overflow_examples():
    big = f16(300.0)
    sq = f16(big * big)                     # 90000 > 65504
    diff = f16(sq - sq)                     # inf − inf = NaN
    e11 = f16(math.exp(11.0))
    e12 = f16(math.exp(12.0))
    # 어텐션 점수 [12, 11, 0] 을 그대로 softmax: exp를 fp16에 저장하면 inf/inf = NaN
    scores = [12.0, 11.0, 0.0]
    ex = [f16(math.exp(s)) for s in scores]
    tot = f16(sum(ex)) if not any(math.isinf(v) for v in ex) else math.inf
    naive = [f16(v / tot) if not (math.isinf(v) and math.isinf(tot)) else math.nan for v in ex]
    # 최댓값을 먼저 빼면(log-sum-exp 요령) 넘치지 않는다
    mx = max(scores)
    ex2 = [f16(math.exp(s - mx)) for s in scores]
    tot2 = f16(sum(ex2))
    stable = [f16(v / tot2) for v in ex2]
    return {"a": 300.0, "a_squared_fp16": sq, "a_squared_bf16": bf16(bf16(300.0) * bf16(300.0)),
            "inf_minus_inf": diff, "exp11": math.exp(11.0), "exp12": math.exp(12.0), "exp11_fp16": e11, "exp12_fp16": e12,
            "exp_threshold": math.log(65504.0),
            "softmax_naive_fp16": naive, "softmax_stable_fp16": stable}
