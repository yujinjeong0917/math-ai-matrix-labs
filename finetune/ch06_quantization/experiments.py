"""파인튜닝 트랙 6장 실험. `uv run python finetune/ch06_quantization/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 숫자 4개 [0.8, -2.0, 0.1, 0.4]를 int8, 균일 4bit, fp4, NF4로 줄였다 되돌리기
E1 튀는 값 하나: 숫자 8개를 한 덩어리로 줄일 때와 4개씩 두 블록으로 줄일 때(균일 4bit)
E2 합성 가중치 65,536개(평균 0, 표준편차 0.02): 방식별 되돌린 오차, 튀는 값을 섞은 스트레스 시험
E3 블록 크기: 32~4096에서 오차와 숫자 하나당 비트
E4 칸 사용률: 균일 4bit와 NF4에서 칸마다 몇 %가 들어가나
E5 NF4 표 다시 만들기: bitsandbytes 방식, 논문 식 (4), 표 값 비교
E6 두 번 양자화: 숫자 하나당 비트와 오차(단순 모형)
E7 7B 가중치 크기: bf16, int8, NF4, NF4 + 두 번 양자화
E8 대조: 무작위 500경우에서 오차 상한과 다시 줄여도 그대로인지

표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch06_quant as Q  # noqa: E402

OUT = HERE / "results" / "ch06.json"
METHODS = ["int8", "int4u", "fp4", "nf4"]


def r(x, nd=6):
    return round(float(x), nd)


def e0_hand():
    xs = [0.8, -2.0, 0.1, 0.4]
    a = Q.absmax(xs)
    out = {"block": xs, "absmax": a, "normalized": [r(x / a) for x in xs]}
    c8, _ = Q.quant_uniform(xs, 127, 4)
    c4, _ = Q.quant_uniform(xs, 7, 4)
    out["int8_codes"] = c8
    out["int4u_codes"] = c4
    nf = Q.nf4_table()
    cn, _ = Q.quant_table(xs, nf, 4)
    out["nf4_codes"] = cn
    out["nf4_picked"] = [r(nf[i], 4) for i in cn]
    for m in METHODS:
        ys = Q.roundtrip(xs, m, 4)
        errs = [abs(y - x) for x, y in zip(xs, ys)]
        out[m] = {"restored": [r(y, 5) for y in ys], "abs_err": [r(e, 5) for e in errs], "sum_abs_err": r(sum(errs), 4)}
    return out


def e1_outlier():
    xs = [0.8, -2.0, 0.1, 0.4, 0.3, -0.5, 20.0, 0.2]
    one = Q.roundtrip(xs, "int4u", None)
    two = Q.roundtrip(xs, "int4u", 4)
    return {"tensor": xs, "one_block_restored": [r(y, 4) for y in one], "two_blocks_restored": [r(y, 4) for y in two],
            "one_block_zeros": sum(1 for x, y in zip(xs, one) if x != 0 and y == 0),
            "two_blocks_zeros": sum(1 for x, y in zip(xs, two) if x != 0 and y == 0),
            "one_block_sum_abs_err": r(sum(abs(y - x) for x, y in zip(xs, one)), 4),
            "two_blocks_sum_abs_err": r(sum(abs(y - x) for x, y in zip(xs, two)), 4)}


def _row(xs, ys):
    return {k: r(v, 6) for k, v in Q.errors(xs, ys).items()}


def e2_gaussian():
    w = Q.synthetic_weights()
    rows = {}
    for m in METHODS:
        for bs in (None, 64):
            rows[f"{m}_{'tensor' if bs is None else 'b64'}"] = _row(w, Q.roundtrip(w, m, bs))
    wo, idx = Q.inject_outliers(w)
    keep = sorted(set(range(len(w))) - set(idx))
    stress = {}
    for m in ("int8", "int4u", "nf4"):
        for bs in (None, 64):
            ys = Q.roundtrip(wo, m, bs)
            stress[f"{m}_{'tensor' if bs is None else 'b64'}"] = _row([wo[i] for i in keep], [ys[i] for i in keep])
    raw = {m: Q.errors(w, Q.roundtrip(w, m, 64))["rmse"] for m in ("nf4", "int4u", "fp4")}  # 반올림 전 값으로 비율을 낸다
    nf, iu, fp = raw["nf4"], raw["int4u"], raw["fp4"]
    return {"n": len(w), "sigma": 0.02, "seed": 6, "rows": rows,
            "nf4_vs_int4u_rmse_cut": r(1 - nf / iu, 6), "nf4_vs_fp4_rmse_cut": r(1 - nf / fp, 6),
            "stress": {"every": 1000, "size": 0.5, "n_outliers": len(idx), "rows": stress}}


def e3_blocksize():
    w = Q.synthetic_weights()
    rows = []
    for bs in Q.load_specs()["bitsandbytes"]["valid_4bit_blocksizes"]:
        rows.append({"blocksize": bs, "bits_per_param": r(Q.bits_per_param(4, bs), 6),
                     "nf4_rel_rmse": r(Q.errors(w, Q.roundtrip(w, "nf4", bs))["rel_rmse"], 4),
                     "int4u_rel_rmse": r(Q.errors(w, Q.roundtrip(w, "int4u", bs))["rel_rmse"], 4)})
    return rows


def e4_usage():
    w = Q.synthetic_weights()
    return {"int4u": [r(v, 4) for v in Q.code_usage(w, "int4u")], "nf4": [r(v, 4) for v in Q.code_usage(w, "nf4")],
            "nf4_table": [r(v, 4) for v in Q.nf4_table()]}


def e5_tables():
    t = Q.nf4_table()
    rec = Q.nf4_bnb_recipe()
    eq4 = Q.nf4_paper_eq4()
    return {"bnb_recipe_max_diff": max(abs(a - b) for a, b in zip(t, rec)),
            "eq4_symmetric": [None if v is None else r(v, 4) for v in eq4],
            "eq4_has_infinite_ends": eq4[0] is None,
            "nf4_positive_count": sum(1 for v in t if v > 0), "nf4_negative_count": sum(1 for v in t if v < 0),
            "nf4_has_zero": 0.0 in t,
            "nf4_max_gap": r(max(b - a for a, b in zip(t, t[1:])), 4), "nf4_min_gap": r(min(b - a for a, b in zip(t, t[1:])), 4),
            "fp4_table": [r(v, 4) for v in Q.fp4_table()], "fp4_distinct": len(Q.fp4_table())}


def e6_double_quant():
    w = Q.synthetic_weights()
    plain = Q.errors(w, Q.roundtrip(w, "nf4", 64))
    dq = Q.errors(w, Q.roundtrip_nf4_dq(w, 64, 256))
    before = Q.bits_per_param(4, 64) - 4
    after = Q.bits_per_param(4, 64, dq_blocksize2=256) - 4
    return {"const_bits_before": before, "const_bits_after": r(after, 6), "reduction": r(before - after, 6),
            "nf4_rel_rmse": r(plain["rel_rmse"], 6), "nf4_dq_rel_rmse": r(dq["rel_rmse"], 6),
            "bits_per_param_nf4": Q.bits_per_param(4, 64), "bits_per_param_nf4_dq": r(Q.bits_per_param(4, 64, dq_blocksize2=256), 6)}


def e7_sizes(n=7e9):
    rows = [("bf16", 16), ("int8_b64", Q.bits_per_param(8, 64)), ("int4_no_consts", 4),
            ("nf4_b64", Q.bits_per_param(4, 64)), ("nf4_b64_dq", Q.bits_per_param(4, 64, dq_blocksize2=256))]
    return {"n_params": n, "rows": [{"name": k, "bits_per_param": r(b, 6), "gb": r(Q.model_gb(n, b), 4)} for k, b in rows]}


def e8_crosscheck(cases=500, seed=8):
    rng = random.Random(seed)
    nf = Q.nf4_table()
    gap = max(b - a for a, b in zip(nf, nf[1:]))
    bad_bound = bad_idem = 0
    for _ in range(cases):
        n = rng.randint(4, 200)
        bs = rng.choice([4, 8, 16, 32, 64])
        xs = [rng.gauss(0, rng.choice([0.01, 0.1, 1.0])) for _ in range(n)]
        for m, qmax in (("int8", 127), ("int4u", 7)):
            ys = Q.roundtrip(xs, m, bs)
            for i, (x, y) in enumerate(zip(xs, ys)):
                a = Q.absmax(xs[(i // bs) * bs:(i // bs) * bs + bs])
                if abs(y - x) > a / (2 * qmax) + 1e-7 * a:
                    bad_bound += 1
            if Q.roundtrip(ys, m, bs) != ys and max(abs(p - q) for p, q in zip(Q.roundtrip(ys, m, bs), ys)) > 1e-6:
                bad_idem += 1
        ys = Q.roundtrip(xs, "nf4", bs)
        for i, (x, y) in enumerate(zip(xs, ys)):
            a = Q.absmax(xs[(i // bs) * bs:(i // bs) * bs + bs])
            if abs(y - x) > a * gap / 2 + 1e-7 * a:
                bad_bound += 1
    return {"cases": cases, "seed": seed, "bound_violations": bad_bound, "idempotence_failures": bad_idem}


def main():
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": Q.SPEC_FILE.name, "checked": "2026-10-11",
                 "rounding": "정수 코드는 round()(가운데는 짝수 쪽), 표는 가장 가까운 값(같으면 앞), absmax는 fp32"},
        "e0_hand": e0_hand(), "e1_outlier": e1_outlier(), "e2_gaussian": e2_gaussian(), "e3_blocksize": e3_blocksize(),
        "e4_usage": e4_usage(), "e5_tables": e5_tables(), "e6_double_quant": e6_double_quant(), "e7_sizes": e7_sizes(),
        "e8_crosscheck": e8_crosscheck(),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
