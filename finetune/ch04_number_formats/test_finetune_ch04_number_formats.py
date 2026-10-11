"""내 데이터로 LLM 파인튜닝하기 4장 검증. `uv run pytest -q finetune/ch04_number_formats` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
numpy·torch 대조 테스트는 설치돼 있을 때만 돈다(importorskip). 나머지는 표준 라이브러리만 쓴다.
"""

import json
import math
import random
from fractions import Fraction as F
from pathlib import Path

import pytest

import ft_ch04_formats as FM
import ft_ch04_scaling as SC

HERE = Path(__file__).parent


# ---------- 출처 파일 ----------

def test_sources_have_url_and_date():
    src = json.loads((HERE / "data" / "sources_2026-10-11.json").read_text(encoding="utf-8"))
    assert src["checked"] == "2026-10-11"
    for f in src["facts"].values():
        assert f["url"].startswith("https://") and f["checked"] == "2026-10-11"


# ---------- 형식 한계값: 비트 수에서 계산한 값이 공식 문서 숫자와 같다 ----------

def test_fp16_limits_match_nvidia_and_python_docs():
    s = FM.spec("fp16")
    assert s["max"] == 65504                          # NVIDIA 가이드 65,504
    assert s["min_subnormal"] == F(1, 2 ** 24)        # NVIDIA 가이드 2^-24
    assert float(s["min_normal"]) == pytest.approx(6.1e-05, rel=1e-3)   # Python 문서 "approximately 6.1e-05"
    assert s["eps"] == F(1, 1024) and s["bytes"] == 2


def test_bf16_has_fp32_range_but_fewer_mantissa_bits():
    b, f = FM.spec("bf16"), FM.spec("fp32")
    assert b["min_normal"] == f["min_normal"] and b["bias"] == f["bias"] == 127
    assert float(b["max"]) == pytest.approx(3.3895e38, rel=1e-4)
    assert b["eps"] == F(1, 128) and f["eps"] == F(1, 2 ** 23)


def test_fp16_boundaries_by_actual_rounding():
    assert FM.f16(65504.0) == 65504.0
    assert FM.f16(65519.99) == 65504.0                # 가운데(65520)보다 작으면 최댓값으로
    assert math.isinf(FM.f16(65520.0))                # 가운데에서 짝수 쪽은 inf
    assert FM.f16(2.0 ** -25) == 0.0                  # 가장 작은 비정규수의 절반은 0(짝수 쪽)
    assert FM.f16(2.0 ** -25 * 1.0001) == 2.0 ** -24
    assert FM.f16(1e-8) == 0.0


# ---------- 비트 분해 손계산 ----------

def test_ten_point_five_bits():
    d16 = FM.decompose("fp16", 10.5)
    assert (d16["sign"], d16["exponent"], d16["mantissa"]) == ("0", "10010", "0101000000")
    assert d16["hex"] == "0x4940" and d16["exp_value"] == 3 and d16["significand"] == 1.3125
    db = FM.decompose("bf16", 10.5)
    assert (db["exponent"], db["mantissa"]) == ("10000010", "0101000") and db["hex"] == "0x4128"
    d32 = FM.decompose("fp32", 10.5)
    assert d32["bits"][:16] == db["bits"]             # bf16 = fp32의 위 16비트
    for d in (d16, db, d32):
        assert d["value_from_fields"] == 10.5


def test_point_one_and_rounding_rule():
    assert FM.f16(0.1) == 0.0999755859375
    assert FM.bf16(0.1) == 0.10009765625
    # 자르기와 가장 가까운 짝수 반올림이 다른 예
    assert FM.bf16(1.012, "trunc") == 1.0078125
    assert FM.bf16(1.012) == 1.015625
    assert FM.bf16_bits(float("nan")) == 0x7FC0


# ---------- 넘침 ----------

def test_overflow_examples():
    o = SC.overflow_examples()
    assert math.isinf(o["a_squared_fp16"]) and math.isnan(o["inf_minus_inf"])
    assert o["a_squared_bf16"] == 90112.0
    assert round(o["exp11"]) == 59874 and round(o["exp12"]) == 162755
    assert o["exp11_fp16"] == 59872.0 and math.isinf(o["exp12_fp16"])
    assert o["exp_threshold"] == pytest.approx(11.0899, abs=1e-4)
    assert math.isnan(o["softmax_naive_fp16"][0])
    assert not any(math.isnan(v) for v in o["softmax_stable_fp16"])
    assert sum(o["softmax_stable_fp16"]) == pytest.approx(1.0, abs=2e-3)


# ---------- 사라짐과 loss scaling ----------

def test_toy_gradient_underflows_and_scaling_rescues():
    w, x, y = 1.0, 0.001, 0.00099
    assert SC.toy_grad(w, x, y) == pytest.approx(1e-8, rel=1e-9)
    assert SC.grad_through_fp16(w, x, y, 1.0)["lost"]
    r10 = SC.grad_through_fp16(w, x, y, 1024.0)
    r16 = SC.grad_through_fp16(w, x, y, 65536.0)
    assert not r10["lost"] and r10["rel_err"] < 0.002
    assert not r16["lost"] and r16["rel_err"] < 0.0005
    # 순서가 중요: fp16에 먼저 저장하고 나서 곱하면 이미 0이다
    assert FM.f16(SC.toy_grad(w, x, y)) * 65536.0 == 0.0


def test_census_shifts_out_of_zero_into_overflow():
    g = SC.synthetic_grads(seed=4)
    c1, c16, c24 = SC.census(g, 1.0), SC.census(g, 2.0 ** 16), SC.census(g, 2.0 ** 24)
    assert c1["zero"] == 2791 and c1["overflow"] == 0
    assert c16["zero"] == 0 and c16["overflow"] == 0
    assert c24["overflow"] == 1599
    assert sum(c1.values()) == 10_000


def test_dynamic_scaler_follows_gradscaler_rules():
    tr = SC.scaler_trace([[2.0, 1e-8]] * 8, growth_interval=3)
    assert [t["scale_used"] for t in tr] == [65536, 32768, 16384, 16384, 16384, 32768, 16384, 16384]
    assert [t["skipped"] for t in tr] == [True, True, False, False, False, True, False, False]
    assert SC.GRADSCALER_DEFAULTS == {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                                      "growth_interval": 2000}
    long = SC.scaler_trace([[2.0, 1e-8]] * 10_000)
    assert sum(t["skipped"] for t in long) == 6
    # 매 스텝 inf(순전파 넘침이나 폭주한 기울기)면 S는 계속 반으로 준다
    assert SC.scaler_trace([[math.inf]] * 17)[-1]["next_scale"] == 0.5


# ---------- 주 가중치 ----------

def test_small_updates_vanish_without_fp32_master():
    assert SC.accumulate(1e-4, 1000, "bf16") == 1.0
    assert SC.accumulate(1e-4, 1000, "fp16") == 1.0
    assert SC.accumulate(1e-4, 1000, "fp32") == pytest.approx(1.1, abs=1e-4)
    m = SC.master_weight_run(1e-4, 1000)
    assert m["copy_low"] == 1.1015625


# ---------- 다른 구현과 대조 ----------

def _fp32_values(n=5000, seed=7):
    rng = random.Random(seed)
    return [FM.f32(rng.choice((-1, 1)) * 10 ** rng.uniform(-12, 6)) for _ in range(n)]


def test_fp16_matches_numpy():
    np = pytest.importorskip("numpy")
    xs = _fp32_values()
    with np.errstate(over="ignore"):
        ref = np.array(xs, dtype=np.float32).astype(np.float16).astype(np.float64).tolist()
    assert all(FM.f16(a) == b for a, b in zip(xs, ref))


def test_bf16_rne_matches_torch():
    torch = pytest.importorskip("torch")
    xs = _fp32_values() + [1.012, 0.1, 10.5, 3.0e38]
    bits = torch.tensor(xs, dtype=torch.float32).to(torch.bfloat16).view(torch.int16).tolist()
    assert all(FM.bf16_bits(a) == (b & 0xFFFF) for a, b in zip(xs, bits))
