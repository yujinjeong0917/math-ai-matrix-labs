"""파인튜닝 트랙 6장 검증. `uv run pytest -q finetune/ch06_quantization` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
GPU·네트워크·API 키 없이 돈다.
"""

import json
from pathlib import Path

import pytest

import ft_ch06_quant as Q

HERE = Path(__file__).parent
HAND = [0.8, -2.0, 0.1, 0.4]


# ---------- 표와 출처 ----------

def test_spec_file_has_date_and_sources():
    s = Q.load_specs()
    assert s["checked"] == "2026-10-11" and Q.SPEC_FILE.name == "specs_2026-10-11.json"
    assert s["bitsandbytes"]["version"] == "0.50.2"
    for url in (s["bitsandbytes"]["url_source"], s["transformers_BitsAndBytesConfig"]["url"], s["qlora_paper"]["url"]):
        assert url.startswith("https://")
    # transformers 문서의 기본값: 4bit 형식은 fp4, 두 번 양자화는 꺼짐
    cfg = s["transformers_BitsAndBytesConfig"]
    assert cfg["bnb_4bit_quant_type"] == "fp4" and cfg["bnb_4bit_use_double_quant"] is False


def test_nf4_table_shape():
    t = Q.nf4_table()
    assert len(t) == 16 and t == sorted(t)
    assert t[0] == -1.0 and t[-1] == 1.0 and 0.0 in t
    assert sum(v > 0 for v in t) == 8 and sum(v < 0 for v in t) == 7   # 비대칭: 양수 쪽이 하나 더 많다


def test_nf4_table_matches_bnb_recipe():
    # bitsandbytes create_normal_map(0.9677083, use_extra_value=True)를 표준 라이브러리로 다시 만들면 표와 float32 반올림 안에서 같다
    assert max(abs(a - b) for a, b in zip(Q.nf4_table(), Q.nf4_bnb_recipe())) < 1e-6


def test_paper_eq4_symmetric_has_infinite_end():
    # 논문 식 (4)를 대칭으로 그대로 쓰면 i=0에서 Q(0) = -무한대가 나와 계산이 안 된다
    eq = Q.nf4_paper_eq4()
    assert eq[0] is None and 0.0 in [round(v, 12) for v in eq if v is not None]


def test_fp4_table_is_not_uniform_int4():
    fp = Q.fp4_table()
    assert len(fp) == 15                                 # 0과 -0이 같은 값이라 서로 다른 값은 15개
    assert pytest.approx(min(v for v in fp if v > 0)) == 0.0625 / 12
    assert fp != Q.uniform_table(7)


# ---------- 첫 화면 손계산 ----------

def test_hand_int8():
    codes, scales = Q.quant_uniform(HAND, 127, 4)
    assert scales == [2.0] and codes == [51, -127, 6, 25]   # 0.4×127=50.8, 0.05×127=6.35, 0.2×127=25.4
    back = Q.dequant_uniform(codes, scales, 127, 4)
    assert [round(v, 4) for v in back] == [0.8031, -2.0, 0.0945, 0.3937]


def test_hand_int4_uniform():
    codes, _ = Q.quant_uniform(HAND, 7, 4)
    assert codes == [3, -7, 0, 1]                            # 2.8, -7, 0.35, 1.4 를 반올림
    back = Q.roundtrip(HAND, "int4u", 4)
    assert [round(v, 4) for v in back] == [0.8571, -2.0, 0.0, 0.2857]


def test_hand_nf4():
    t = Q.nf4_table()
    codes, _ = Q.quant_table(HAND, t, 4)
    assert [round(t[i], 4) for i in codes] == [0.4407, -1.0, 0.0796, 0.1609]
    back = Q.roundtrip(HAND, "nf4", 4)
    assert [round(v, 4) for v in back] == [0.8814, -2.0, 0.1592, 0.3219]
    # 이 네 숫자에서는 NF4가 0.8을 균일 4bit보다 더 틀린다. 합으로는 NF4가 작다
    e_nf = [abs(a - b) for a, b in zip(HAND, back)]
    e_iu = [abs(a - b) for a, b in zip(HAND, Q.roundtrip(HAND, "int4u", 4))]
    assert e_nf[0] > e_iu[0] and sum(e_nf) < sum(e_iu)


def test_round_half_to_even_rule():
    assert round(2.5) == 2 and round(3.5) == 4
    assert Q.quant_uniform([2.5, 7.0], 7, None)[0] == [2, 7]


# ---------- 블록 ----------

def test_outlier_only_hurts_its_block():
    xs = [0.8, -2.0, 0.1, 0.4, 0.3, -0.5, 20.0, 0.2]
    one = Q.roundtrip(xs, "int4u", None)
    two = Q.roundtrip(xs, "int4u", 4)
    assert [round(v, 4) for v in one] == [0.0, -2.8571, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0]
    assert [round(v, 4) for v in two[:4]] == [0.8571, -2.0, 0.0, 0.2857]   # 앞 블록은 손계산 그대로


def test_blocks_beat_one_scale_on_synthetic_weights():
    w = Q.synthetic_weights()
    for m in ("int8", "int4u", "fp4", "nf4"):
        assert Q.errors(w, Q.roundtrip(w, m, 64))["rmse"] < Q.errors(w, Q.roundtrip(w, m, None))["rmse"]


def test_nf4_beats_other_4bit_at_block64():
    w = Q.synthetic_weights()
    e = {m: Q.errors(w, Q.roundtrip(w, m, 64))["rmse"] for m in ("int4u", "fp4", "nf4")}
    assert e["nf4"] < e["int4u"] < e["fp4"]


# ---------- 비트와 크기 ----------

def test_bits_per_param_matches_paper():
    assert Q.bits_per_param(4, 64) - 4 == 0.5                       # 32/64
    dq = Q.bits_per_param(4, 64, dq_blocksize2=256) - 4
    assert dq == 8 / 64 + 32 / (64 * 256)
    assert round(dq, 3) == 0.127 and round(0.5 - dq, 3) == 0.373   # 논문의 반올림 값


def test_model_sizes_7b():
    assert Q.model_gb(7e9, 16) == 14.0
    assert Q.model_gb(7e9, 4) == 3.5
    assert Q.model_gb(7e9, 4.5) == pytest.approx(3.9375)
    assert Q.model_gb(7e9, Q.bits_per_param(4, 64, dq_blocksize2=256)) == pytest.approx(3.6111, abs=1e-4)


def test_double_quant_barely_changes_error():
    w = Q.synthetic_weights()
    a = Q.errors(w, Q.roundtrip(w, "nf4", 64))["rel_rmse"]
    b = Q.errors(w, Q.roundtrip_nf4_dq(w))["rel_rmse"]
    assert abs(a - b) < 1e-3


# ---------- 오차 상한 ----------

@pytest.mark.parametrize("qmax", [127, 7])
def test_uniform_error_within_half_step(qmax):
    import random
    rng = random.Random(1)
    xs = [rng.gauss(0, 0.05) for _ in range(256)]
    ys = Q.dequant_uniform(*Q.quant_uniform(xs, qmax, 32), qmax, 32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        a = Q.absmax(xs[(i // 32) * 32:(i // 32) * 32 + 32])
        assert abs(y - x) <= a / (2 * qmax) * (1 + 1e-6)


# ---------- 결과 파일 ----------

def test_results_file_matches_recomputation():
    res = json.loads((HERE / "results" / "ch06.json").read_text(encoding="utf-8"))
    assert res["e0_hand"]["int8_codes"] == [51, -127, 6, 25]
    w = Q.synthetic_weights()
    assert res["e2_gaussian"]["rows"]["nf4_b64"]["rmse"] == round(Q.errors(w, Q.roundtrip(w, "nf4", 64))["rmse"], 6)
    assert res["e8_crosscheck"]["bound_violations"] == 0 and res["e8_crosscheck"]["idempotence_failures"] == 0
    assert res["e6_double_quant"]["const_bits_after"] == 0.126953
