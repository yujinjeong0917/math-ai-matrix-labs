"""14장 실험. `python experiments.py` 로 실행하면 results/results.json 을 쓰고 표를 출력한다.

A. 파라미터·FLOP 세기: Mixtral 8x7B 설정표(Jiang 등 2024, Table 1)로 총·활성 파라미터를 식으로 센다. 작은 모델은 torch의 FLOP 계수기로 대조한다.
B. 장난감 학습: 8군집 합성 회귀에서 밀집(활성 계산 같은 것, 총 파라미터 같은 것)과 MoE(보조 손실 α 여섯 값)를 시드 5개로 비교한다.
C. 전문가 용량: B의 라우팅 결과로 용량 계수별 넘치는 토큰 비율을 센다.
D. KV cache 바이트: Mixtral 설정에서 KV 헤드 수를 32(MHA), 8(GQA), 1(MQA)로 바꿨을 때의 산술.

숫자는 모두 가상의 합성 데이터와 공개된 설정표에서 나온다. 실제 모델의 성능을 재현한 것이 아니다.
"""

import json
import pathlib
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import moe_np as mn
import moe_torch as mt

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# Jiang 등(2024) Table 1
MIXTRAL = {"d": 4096, "L": 32, "dh": 128, "ff": 14336, "h": 32, "kv": 8, "V": 32000}
E = 8  # 전문가 수. 바꾸면 모델 생성, 쏠림 배수, 용량 계산이 함께 바뀐다
SEEDS = range(5)
ALPHAS = [0.0, 1e-2, 1e-1, 1.0, 3.0, 10.0]


def part_a():
    tot, act = mn.model_params(MIXTRAL, E=8, k=2)
    dense_tot, _ = mn.model_params(MIXTRAL, E=1, k=1)
    # 작은 모델: 밀집 h=16 한 번, MoE E=8 k=1 한 번의 순전파 FLOP을 torch가 직접 센다
    x = torch.randn(256, mt.D_IN)
    flops = {}
    for name, m in {"dense_h16": mt.Dense(16), "dense_h128": mt.Dense(128), "moe_e8_k1_h16": mt.MoE(E, 1, 16),
                    "moe_e8_k2_h8": mt.MoE(E, 2, 8)}.items():
        with FlopCounterMode(display=False) as fc:
            m(x)
        flops[name] = fc.get_total_flops()
    params = {n: sum(p.numel() for p in m.parameters()) for n, m in
              {"dense_h16": mt.Dense(16), "dense_h128": mt.Dense(128), "moe_e8_k1_h16": mt.MoE(E, 1, 16),
               "moe_e8_k2_h8": mt.MoE(E, 2, 8)}.items()}
    return {"mixtral_total": tot, "mixtral_active": act, "mixtral_dense_equiv_total": dense_tot,
            "toy_flops_256tok": flops, "toy_params": params}


def part_b_c():
    rows = {}
    caps = {}
    for seed in SEEDS:
        data = mt.make_data(seed)
        specs = {"dense_h16": (lambda: mt.Dense(16), 0.0), "dense_h128": (lambda: mt.Dense(128), 0.0)}
        for a in ALPHAS:
            # Switch식: 전체 softmax 확률을 게이트로 곱한다(k=1). 라우터가 과제 기울기를 받는다
            specs[f"switch_k1_alpha_{a:g}"] = ((lambda: mt.MoE(E, 1, 16, gate="switch")), a)
        # 흔한 실패: k=1인데 상위 1개에만 다시 softmax하면 게이트가 항상 1이라 라우터가 과제 기울기를 못 받는다
        specs["renorm_k1_alpha_0"] = ((lambda: mt.MoE(E, 1, 16, gate="renorm")), 0.0)
        # Mixtral식: k=2, 상위 2개에 softmax. 전문가는 h=8이라 토큰당 활성 은닉 크기는 8 x 2 = 16(밀집 h=16과 같다)
        specs["renorm_k2_alpha_0"] = ((lambda: mt.MoE(E, 2, 8, gate="renorm")), 0.0)
        specs["renorm_k2_alpha_0.01"] = ((lambda: mt.MoE(E, 2, 8, gate="renorm")), 1e-2)
        for name, (make, a) in specs.items():
            torch.manual_seed(seed)
            m = make()
            w0 = m.router.weight.detach().clone() if hasattr(m, "router") else None
            mse, f = mt.train(m, data, alpha=a, seed=seed)
            rec = {"mse": mse}
            if f is not None:
                rec.update({"f": f.tolist(), "max_over_mean": float(f.max() * E), "unused": int((f < 0.01).sum()),
                            "router_move": float((m.router.weight.detach() - w0).norm())})
                # C: 같은 라우팅으로 용량 계수별 버려지는 토큰
                with torch.no_grad():
                    _, _, top = m(data[1][0])
                assign = top[:, 0].numpy()
                T = len(assign)
                for cf in (1.0, 1.25, 1.5, 2.0):
                    cap = mn.expert_capacity(T, E, cf)
                    caps.setdefault(name, {}).setdefault(str(cf), []).append(mn.dropped_tokens(assign, E, cap) / T)
            rows.setdefault(name, []).append(rec)
    return rows, caps


def part_d():
    out = {}
    for name, kv in {"MHA": 32, "GQA(8)": 8, "MQA": 1}.items():
        per = mn.kv_bytes_per_token(MIXTRAL["L"], kv, MIXTRAL["dh"], 2)
        out[name] = {"per_token_bytes": per, "at_32768_GiB": per * 32768 / 2**30}
    return out


def summarize(rows):
    s = {}
    for name, recs in rows.items():
        m = [r["mse"] for r in recs]
        s[name] = {"mse_mean": float(np.mean(m)), "mse_min": float(np.min(m)), "mse_max": float(np.max(m))}
        if "max_over_mean" in recs[0]:
            mm = [r["max_over_mean"] for r in recs]
            s[name].update({"max_over_mean_min": float(np.min(mm)), "max_over_mean_max": float(np.max(mm)),
                            "unused_per_seed": [r["unused"] for r in recs],
                            "router_move_mean": float(np.mean([r["router_move"] for r in recs]))})
    return s


if __name__ == "__main__":
    A = part_a()
    rows, caps = part_b_c()
    D = part_d()
    res = {"A": A, "B_summary": summarize(rows), "B_raw": rows,
           "C_dropped_fraction_mean": {n: {cf: float(np.mean(v)) for cf, v in d.items()} for n, d in caps.items()},
           "D": D}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("A", "B_summary", "C_dropped_fraction_mean", "D")}, indent=1, ensure_ascii=False))
