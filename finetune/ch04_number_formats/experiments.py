"""내 데이터로 LLM 파인튜닝하기 4장 실험. `uv run python finetune/ch04_number_formats/experiments.py` 로 실행한다.

E0 비트 분해 손계산: 10.5를 fp32·fp16·bf16 칸으로, 0.1과 1.012를 세 형식으로
E1 형식별 한계값: 비트 수에서 최댓값·가장 작은 정규수·비정규수·eps·바이트
E2 넘침: 300 × 300, inf − inf, exp(11)·exp(12), 어텐션 점수 softmax(그대로 / 최댓값 빼고)
E3 사라짐: 장난감 손실 ½(wx − y)²의 기울기 1e-8을 fp16에, loss scaling 1·1024·65536
E4 가상 기울기 10,000개를 scale별로 fp16에 저장했을 때 0·비정규수·정규수·넘침 개수
E5 동적 스케일러 손 추적(growth_interval을 3으로 줄인 판)과 기본값(2000)으로 10,000스텝
E6 작은 업데이트가 사라지는 문제: 1에 1e-4를 1,000번 더하기, fp32 주 가중치
E7 메모리: 파라미터 10억 개를 형식별로 저장할 때(가중치만)
E8 대조: 무작위 fp32 값 20,000개에서 struct 'e' ↔ numpy.float16, 우리 bf16 ↔ torch.bfloat16(설치돼 있을 때)

표준 라이브러리만으로 E0~E7을 계산한다. 고정 시드라 결과는 매번 같다.
"""

import json
import math
import platform
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch04_formats as FM  # noqa: E402
import ft_ch04_scaling as SC  # noqa: E402

OUT = HERE / "results" / "ch04.json"
SEED = 4
SOURCES = HERE / "data" / "sources_2026-10-11.json"


def fl(x):
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return x


def e0_bits():
    rows = []
    for fmt in ("fp32", "fp16", "bf16"):
        d = FM.decompose(fmt, 10.5)
        rows.append({k: d[k] for k in ("format", "bits", "sign", "exponent", "mantissa", "hex",
                                        "exp_field", "exp_value", "significand", "value_from_fields")})
    stored = []
    for x in (0.1, 1.012, 300.0, 1e-8, 3e-5):
        stored.append({"x": x, "fp32": FM.f32(x), "fp16": fl(FM.f16(x)), "bf16_rne": FM.bf16(x),
                       "bf16_trunc": FM.bf16(x, "trunc"), "fp16_kind": FM.classify("fp16", x)})
    return {"ten_point_five": rows, "stored": stored,
            "bf16_spacing_near_1": FM.spacing_at("bf16", 1.0), "fp16_spacing_near_1": FM.spacing_at("fp16", 1.0),
            "fp32_spacing_near_1": FM.spacing_at("fp32", 1.0)}


def e1_specs():
    out = {}
    for fmt in ("fp32", "fp16", "bf16"):
        s = FM.spec(fmt)
        out[fmt] = {k: (v if isinstance(v, int) else float(v)) for k, v in s.items()}
    # 경계 확인(struct로 실제로 반올림해 본다)
    out["fp16_boundary"] = {"65519.99": FM.f16(65519.99), "65520": fl(FM.f16(65520.0)),
                            "2^-25": FM.f16(2.0 ** -25), "2^-25*1.0001": FM.f16(2.0 ** -25 * 1.0001),
                            "2^-24": FM.f16(2.0 ** -24)}
    return out


def e2_overflow():
    o = SC.overflow_examples()
    return {k: ([fl(v) for v in val] if isinstance(val, list) else fl(val)) for k, val in o.items()}


def e3_underflow():
    w, x, y = 1.0, 0.001, 0.00099
    rows = []
    for s in (1.0, 1024.0, 65536.0):
        r = SC.grad_through_fp16(w, x, y, s)
        rows.append({k: fl(v) for k, v in r.items()})
    return {"w": w, "x": x, "y": y, "residual": w * x - y, "true_grad": SC.toy_grad(w, x, y), "rows": rows,
            "wrong_order_scale_after_store": FM.f16(SC.toy_grad(w, x, y)) * 65536.0}


def e4_census():
    g = SC.synthetic_grads(seed=SEED)
    rows = []
    for s in (1.0, 2.0 ** 10, 2.0 ** 16, 2.0 ** 19, 2.0 ** 20, 2.0 ** 24):
        c = SC.census(g, s)
        rows.append({"scale": s, **c, "zero_pct": round(100 * c["zero"] / len(g), 2),
                     "overflow_pct": round(100 * c["overflow"] / len(g), 2)})
    return {"n": len(g), "range": "1e-10 ~ 1e-1 로그 균등(가상)", "seed": SEED, "rows": rows}


def e5_scaler():
    steps = [[2.0, 1e-8]] * 8
    tr = SC.scaler_trace(steps, growth_interval=3)
    hand = [{"step": i + 1, "scale_used": t["scale_used"], "max_scaled": t["max_scaled"],
             "skipped": t["skipped"], "next_scale": t["next_scale"],
             "small_grad_back": None if t["unscaled"] is None else t["unscaled"][1]} for i, t in enumerate(tr)]
    long = SC.scaler_trace([[2.0, 1e-8]] * 10_000)
    skips = sum(t["skipped"] for t in long)
    allinf = SC.scaler_trace([[math.inf]] * 17)
    return {"always_inf_17_steps_final_scale": allinf[-1]["next_scale"],
            "grads_each_step": [2.0, 1e-8], "growth_interval_demo": 3, "hand": hand,
            "default_10000": {"skipped": skips, "skip_pct": round(100 * skips / 10_000, 3),
                              "final_scale": long[-1]["next_scale"],
                              "first_three": [(t["scale_used"], t["skipped"]) for t in long[:3]]}}


def e6_master():
    u, n = 1e-4, 1000
    return {"update": u, "steps": n, "ideal": 1.0 + u * n,
            "fp32": SC.accumulate(u, n, "fp32"), "fp16": SC.accumulate(u, n, "fp16"),
            "bf16": SC.accumulate(u, n, "bf16"),
            "master_bf16": SC.master_weight_run(u, n, low="bf16"),
            "master_fp16": SC.master_weight_run(u, n, low="fp16"),
            "half_spacing_bf16": FM.spacing_at("bf16", 1.0) / 2, "half_spacing_fp16": FM.spacing_at("fp16", 1.0) / 2}


def e7_memory():
    p = 1_000_000_000
    return {"params": p, "GB": {fmt: p * FM.spec(fmt)["bytes"] / 1e9 for fmt in ("fp32", "fp16", "bf16")}}


def e8_crosscheck():
    rng = random.Random(SEED)
    xs = []
    for _ in range(20_000):
        v = rng.choice((-1, 1)) * 10 ** rng.uniform(-12, 6)
        xs.append(FM.f32(v))   # fp32로 정확히 표현되는 값만 쓴다
    xs += [65504.0, 65519.0, 65520.0, 2.0 ** -24, 2.0 ** -25, 1.012, 0.1]
    xs = [FM.f32(v) for v in xs]
    res = {"n": len(xs), "numpy_fp16_mismatch": None, "torch_bf16_mismatch": None, "trunc_vs_rne_differ": 0}
    res["trunc_vs_rne_differ"] = sum(FM.bf16_bits(v) != FM.bf16_bits(v, "trunc") for v in xs)
    try:
        import numpy as np
        with np.errstate(over="ignore"):
            ref = np.array(xs, dtype=np.float32).astype(np.float16).astype(np.float64)
        res["numpy_fp16_mismatch"] = int(sum(1 for a, b in zip(xs, ref) if FM.f16(a) != b and not (math.isnan(b))))
        res["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import torch
        t = torch.tensor(xs, dtype=torch.float32).to(torch.bfloat16)
        bits = t.view(torch.int16).tolist()
        res["torch_bf16_mismatch"] = int(sum(1 for a, b in zip(xs, bits) if FM.bf16_bits(a) != (b & 0xFFFF)))
        res["torch_version"] = torch.__version__
    except ImportError:
        pass
    return res


def main():
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": SEED, "sources": SOURCES.name, "checked": src["checked"]},
        "e0_bits": e0_bits(),
        "e1_specs": e1_specs(),
        "e2_overflow": e2_overflow(),
        "e3_underflow": e3_underflow(),
        "e4_census": e4_census(),
        "e5_scaler": e5_scaler(),
        "e6_master": e6_master(),
        "e7_memory": e7_memory(),
        "e8_crosscheck": e8_crosscheck(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
