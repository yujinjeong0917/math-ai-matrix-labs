"""fp32 · fp16 · bf16 숫자 형식을 표준 라이브러리만으로 재현한다.

흉내 규칙
- 파이썬 float는 fp64다. 연산 한 번을 fp64로 하고 목표 형식으로 한 번 반올림한다.
  fp64는 fp32·fp16·bf16보다 가수 비트가 충분히 많아서, 덧셈·곱셈·나눗셈 한 번의 결과는
  이렇게 해도 목표 형식에서 바로 계산한 것과 같게 반올림된다(이중 반올림 문제가 생기지 않는 조건).
- fp32: struct 'f', fp16: struct 'e' (둘 다 IEEE 754, 가장 가까운 값으로, 딱 가운데면 짝수 쪽).
  struct는 너무 큰 값을 넣으면 OverflowError를 내지만, GPU 연산은 inf를 낸다. 그래서 inf로 바꾼다.
- bf16: fp32 비트의 위 16비트. 반올림은 PyTorch c10 round_to_nearest_even과 같은 식
  (NaN이면 0x7FC0, 아니면 u + ((u >> 16) & 1) + 0x7FFF 를 16비트 오른쪽으로 민다).
  비교용으로 그냥 자르기(truncate)도 둔다.
"""

import math
import struct
from fractions import Fraction as F

# 형식 이름: (지수 비트, 가수 비트). 부호 비트는 모두 1개.
FORMATS = {"fp32": (8, 23), "fp16": (5, 10), "bf16": (8, 7)}


# ---------- 형식의 한계값: 비트 수만으로 계산 ----------

def spec(fmt):
    """비트 수에서 최댓값·가장 작은 정규수·가장 작은 비정규수·eps를 정확한 분수로 계산한다."""
    e, m = FORMATS[fmt]
    bias = 2 ** (e - 1) - 1
    return {
        "bits": 1 + e + m,
        "exp_bits": e,
        "mant_bits": m,
        "bias": bias,
        "max": (2 - F(1, 2 ** m)) * F(2) ** bias,
        "min_normal": F(1, 2 ** (bias - 1)),
        "min_subnormal": F(1, 2 ** (bias - 1 + m)),
        "eps": F(1, 2 ** m),          # 1과 그다음 수의 간격
        "unit_roundoff": F(1, 2 ** (m + 1)),  # 반올림 한 번의 상대 오차 상한 u = eps/2
        "bytes": (1 + e + m) // 8,
    }


# ---------- 반올림 ----------

def _pack(fmtchar, x):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return struct.unpack("<" + fmtchar, struct.pack("<" + fmtchar, x))[0]
    try:
        return struct.unpack("<" + fmtchar, struct.pack("<" + fmtchar, x))[0]
    except (OverflowError, struct.error):
        return math.copysign(math.inf, x)


def f32(x):
    """fp32로 반올림. 넘치면 ±inf."""
    return _pack("f", x)


def f16(x):
    """fp16으로 반올림(struct 'e'). 넘치면 ±inf. 너무 작으면 0."""
    return _pack("e", x)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def f16_bits(x):
    return struct.unpack("<H", struct.pack("<e", f16(x)))[0]


def bf16_bits(x, mode="rne"):
    """fp32 비트에서 bf16 16비트를 만든다. mode='rne'(PyTorch 기본) 또는 'trunc'(그냥 자르기)."""
    u = f32_bits(x)
    if mode == "trunc":
        return u >> 16
    if math.isnan(f32(x)):
        return 0x7FC0
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) >> 16) & 0xFFFF


def bf16_from_bits(b):
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


def bf16(x, mode="rne"):
    return bf16_from_bits(bf16_bits(x, mode))


ROUND = {"fp32": f32, "fp16": f16, "bf16": bf16}


def to(fmt, x):
    return ROUND[fmt](x)


def bits_of(fmt, x):
    if fmt == "fp32":
        return f32_bits(x)
    if fmt == "fp16":
        return f16_bits(x)
    return bf16_bits(x)


# ---------- 비트 분해 ----------

def decompose(fmt, x):
    """값을 그 형식으로 반올림한 뒤 부호·지수·가수 칸으로 나누고, 칸에서 값을 다시 계산한다."""
    e, m = FORMATS[fmt]
    bias = 2 ** (e - 1) - 1
    b = bits_of(fmt, x)
    sign = b >> (e + m)
    exp_field = (b >> m) & ((1 << e) - 1)
    mant_field = b & ((1 << m) - 1)
    s = f"{b:0{1 + e + m}b}"
    if exp_field == (1 << e) - 1:
        kind = "nan" if mant_field else "inf"
        value = math.nan if mant_field else math.copysign(math.inf, -1 if sign else 1)
        exp_val = None
        frac = None
    elif exp_field == 0:
        kind = "zero" if mant_field == 0 else "subnormal"
        exp_val = 1 - bias
        frac = F(mant_field, 2 ** m)             # 0.가수
        value = float((-1) ** sign * frac * F(2) ** exp_val)
    else:
        kind = "normal"
        exp_val = exp_field - bias
        frac = 1 + F(mant_field, 2 ** m)         # 1.가수
        value = float((-1) ** sign * frac * F(2) ** exp_val)
    return {
        "format": fmt, "input": x, "stored": to(fmt, x), "kind": kind,
        "bits": s, "sign": s[0], "exponent": s[1:1 + e], "mantissa": s[1 + e:],
        "hex": f"0x{b:0{(1 + e + m) // 4}X}",
        "exp_field": exp_field, "exp_value": exp_val,
        "significand": None if frac is None else float(frac),
        "value_from_fields": value,
    }


def classify(fmt, x):
    return decompose(fmt, x)["kind"]


def spacing_at(fmt, x):
    """x(정규수) 근처에서 이웃한 두 수의 간격 = 2^(지수) × eps."""
    d = decompose(fmt, x)
    _, m = FORMATS[fmt]
    return float(F(2) ** d["exp_value"] / 2 ** m)
