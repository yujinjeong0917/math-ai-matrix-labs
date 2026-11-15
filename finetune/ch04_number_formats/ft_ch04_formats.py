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
# 형식 이름 -> (지수 비트, 가수 비트). 세 형식 모두 부호 1비트. bf16은 fp32와 지수 비트가 같고 가수가 짧아 범위는 같고 정밀도는 낮다
FORMATS = {"fp32": (8, 23), "fp16": (5, 10), "bf16": (8, 7)}


# ---------- 형식의 한계값: 비트 수만으로 계산 ----------

def spec(fmt):
    """비트 수에서 최댓값·가장 작은 정규수·가장 작은 비정규수·eps를 정확한 분수로 계산한다."""
    e, m = FORMATS[fmt]
    # 지수 편향: 저장된 지수 칸에서 이 값을 빼 실제 지수를 얻는다(fp16 15, bf16·fp32 127)
    bias = 2 ** (e - 1) - 1
    return {
        "bits": 1 + e + m,
        "exp_bits": e,
        "mant_bits": m,
        "bias": bias,
        # 최댓값 = (2 - 2^-m) x 2^bias. 가수 칸이 모두 1인 정규수 중 가장 큰 지수. Fraction으로 계산해 반올림 오차가 없다
        "max": (2 - F(1, 2 ** m)) * F(2) ** bias,
        "min_normal": F(1, 2 ** (bias - 1)),
        # 가장 작은 비정규수 = 2^(1 - bias - m). 지수 칸이 0이고 가수의 맨 끝 비트만 1
        "min_subnormal": F(1, 2 ** (bias - 1 + m)),
        "eps": F(1, 2 ** m),          # 1과 그다음 수의 간격
        # 한 번 반올림할 때 상대 오차의 상한 u = eps/2
        "unit_roundoff": F(1, 2 ** (m + 1)),  # 반올림 한 번의 상대 오차 상한 u = eps/2
        "bytes": (1 + e + m) // 8,
    }


# ---------- 반올림 ----------

# struct로 형식에 맞게 반올림한다. NaN, inf는 그대로 통과시키고, 범위를 넘는 값은 OverflowError 대신 +-inf를 돌려준다(GPU 연산과 같게)
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
    # bf16은 fp32 비트의 위 16비트. 먼저 fp32로 반올림한 뒤 그 비트를 쓴다
    u = f32_bits(x)
    if mode == "trunc":
        return u >> 16
    if math.isnan(f32(x)):
        return 0x7FC0
    # 가장 가까운 짝수 쪽 반올림(RNE) 식: 하위 16비트가 딱 절반이면 위 16비트의 마지막 비트가 홀수일 때만 올린다. 이 계산은 NaN이 아닌 값에만 쓴다
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
    # 지수 칸이 모두 1이면 inf(가수 0) 또는 NaN(가수 0 아님)
    if exp_field == (1 << e) - 1:
        kind = "nan" if mant_field else "inf"
        value = math.nan if mant_field else math.copysign(math.inf, -1 if sign else 1)
        exp_val = None
        frac = None
    # 지수 칸이 0이면 0 또는 비정규수. 암묵적 1이 없어 가수는 0.xxx이고 지수는 1 - bias로 고정
    elif exp_field == 0:
        kind = "zero" if mant_field == 0 else "subnormal"
        exp_val = 1 - bias
        frac = F(mant_field, 2 ** m)             # 0.가수
        value = float((-1) ** sign * frac * F(2) ** exp_val)
    else:
        kind = "normal"
        exp_val = exp_field - bias
        # 정규수는 암묵적인 앞자리 1을 더해 1.xxx로 읽는다
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
    # 이웃한 두 수의 간격 = 2^(지수) x eps = 2^(지수 - 가수 비트). 정규수에서만 맞다
    return float(F(2) ** d["exp_value"] / 2 ** m)
