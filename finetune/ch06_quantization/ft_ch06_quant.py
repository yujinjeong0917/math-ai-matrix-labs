"""파인튜닝 트랙 6장: 최소한의 양자화.

absmax int8, 블록별 4bit(가르치기용 균일 4bit, bitsandbytes fp4), NF4로 숫자를 줄였다가 되돌리고
오차를 잰다. 표준 라이브러리만 쓴다.

반올림 규칙: 정수 코드로 바꿀 때는 파이썬 round()를 쓴다. 딱 가운데(.5)는 짝수 쪽으로 간다
(round(2.5) == 2). 표(NF4, fp4)에서 고를 때는 가장 가까운 값을 고르고, 거리가 같으면 표에서 앞(작은 값)을 고른다.
블록마다 저장하는 absmax(절댓값 최대)는 fp32로 저장한다고 보고 struct 'f'로 한 번 거쳐 둔다.
"""

import json
import math
import random
import struct
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).parent
SPEC_FILE = HERE / "data" / "specs_2026-10-11.json"


def load_specs():
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def f32(x):
    """fp32로 저장했다가 읽은 값(4바이트 단정밀도)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------- 표 ----------

def nf4_table():
    return list(load_specs()["bitsandbytes"]["nf4"])


def fp4_table():
    """bitsandbytes fp4 표를 값 순서로 정렬하고 최댓값 12로 나눈 것. 0과 -0은 하나로 친다."""
    raw = load_specs()["bitsandbytes"]["fp4_raw_bit_order"]
    vals = sorted({v / 12.0 for v in raw})
    return vals


def uniform_table(qmax):
    """가르치기용 균일 표: -qmax..qmax를 qmax로 나눈 값. int4라면 qmax=7로 15칸을 쓴다."""
    return [k / qmax for k in range(-qmax, qmax + 1)]


def nf4_bnb_recipe(offset=0.9677083):
    """bitsandbytes create_normal_map(offset, use_extra_value=True)를 표준 라이브러리로 다시 만든다.

    torch.linspace(offset, 0.5, 9)[:-1] 의 양수 쪽 8개, linspace(offset, 0.5, 8)[:-1] 의 음수 쪽 7개, 0 하나.
    원본은 linspace를 float32로 만들고 결과를 float32 텐서에 담아 최댓값으로 나눈다. 여기서도 같은 자리에서 f32를 거친다.
    """
    nd = NormalDist()

    def linspace(a, b, n):
        return [f32(a + (b - a) * i / (n - 1)) for i in range(n)]

    pos = [nd.inv_cdf(p) for p in linspace(offset, 0.5, 9)[:-1]]
    neg = [-nd.inv_cdf(p) for p in linspace(offset, 0.5, 8)[:-1]]
    vals = sorted([f32(v) for v in pos + neg] + [0.0])
    m = max(vals)
    return [f32(v / m) for v in vals]


def nf4_paper_eq4(k=4):
    """QLoRA 논문 식 (4)를 글자 그대로 옮긴 대칭 버전: q_i = (Q(i/(2^k+1)) + Q((i+1)/(2^k+1)))/2, i=0..2^k-1.

    i=0이면 Q(0) = -무한대라 그대로는 계산되지 않는다. 그래서 이 함수는 끝의 두 칸을 계산하지 못한다는 사실을
    돌려준다(None). 논문은 비대칭 버전(음수 2^(k-1)개, 양수 2^(k-1)+1개)을 쓰고 정확한 값은 부록 E에 둔다.
    """
    nd = NormalDist()
    n = 2 ** k + 1
    out = []
    for i in range(2 ** k):
        a, b = i / n, (i + 1) / n
        if a <= 0 or b >= 1:
            out.append(None)
        else:
            out.append((nd.inv_cdf(a) + nd.inv_cdf(b)) / 2)
    return out


# ---------- 줄이고 되돌리기 ----------

def nearest_index(v, table):
    best, bi = None, 0
    for i, t in enumerate(table):
        d = abs(v - t)
        if best is None or d < best:
            best, bi = d, i
    return bi


def blocks_of(xs, blocksize):
    if blocksize is None:
        return [list(xs)]
    return [list(xs[i:i + blocksize]) for i in range(0, len(xs), blocksize)]


def absmax(block):
    return f32(max(abs(x) for x in block))


def quant_uniform(xs, qmax, blocksize=None):
    """absmax 균일 양자화. 블록마다 a = max|x|, 코드 = round(x / a * qmax). 돌려주는 것: (코드 목록, absmax 목록)."""
    codes, scales = [], []
    for b in blocks_of(xs, blocksize):
        a = absmax(b)
        scales.append(a)
        for x in b:
            c = 0 if a == 0 else round(x / a * qmax)
            codes.append(max(-qmax, min(qmax, c)))
    return codes, scales


def dequant_uniform(codes, scales, qmax, blocksize=None):
    n = len(codes)
    bs = n if blocksize is None else blocksize
    return [codes[i] / qmax * scales[i // bs] for i in range(n)]


def quant_table(xs, table, blocksize=None):
    """표 양자화(NF4, fp4). 블록마다 x / a 를 표에서 가장 가까운 값의 번호(0..15)로 바꾼다."""
    codes, scales = [], []
    for b in blocks_of(xs, blocksize):
        a = absmax(b)
        scales.append(a)
        for x in b:
            codes.append(nearest_index(0.0 if a == 0 else x / a, table))
    return codes, scales


def dequant_table(codes, scales, table, blocksize=None):
    n = len(codes)
    bs = n if blocksize is None else blocksize
    return [table[codes[i]] * scales[i // bs] for i in range(n)]


def roundtrip(xs, method, blocksize=None):
    """method: 'int8', 'int4u'(가르치기용 균일 4bit), 'fp4', 'nf4'. 되돌린 값 목록을 돌려준다."""
    if method == "int8":
        c, s = quant_uniform(xs, 127, blocksize)
        return dequant_uniform(c, s, 127, blocksize)
    if method == "int4u":
        c, s = quant_uniform(xs, 7, blocksize)
        return dequant_uniform(c, s, 7, blocksize)
    table = {"fp4": fp4_table, "nf4": nf4_table}[method]()
    c, s = quant_table(xs, table, blocksize)
    return dequant_table(c, s, table, blocksize)


# ---------- 두 번 양자화(Double Quantization) 단순 모형 ----------

def double_quant_scales(scales, blocksize2=256):
    """absmax 목록을 다시 줄인다(단순 모형). 평균을 빼고 256개씩 균일 int8(±127)로 줄였다가 되돌린다.

    bitsandbytes는 이 단계에서 균일 int8이 아니라 8비트 동적 표(quantize_blockwise)를 쓴다. 여기서는 원리만 본다.
    """
    m = f32(sum(scales) / len(scales))
    d = [s - m for s in scales]
    c, s2 = quant_uniform(d, 127, blocksize2)
    back = dequant_uniform(c, s2, 127, blocksize2)
    return [m + v for v in back]


def roundtrip_nf4_dq(xs, blocksize=64, blocksize2=256):
    table = nf4_table()
    c, s = quant_table(xs, table, blocksize)
    s_dq = double_quant_scales(s, blocksize2)
    return dequant_table(c, s_dq, table, blocksize)


# ---------- 크기 ----------

def bits_per_param(code_bits, blocksize, const_bits=32, dq_blocksize2=None, dq_const_bits=8):
    """숫자 하나당 비트 수. 코드 비트 + 블록 상수 몫. 두 번 양자화면 상수가 8비트가 되고 2단계 fp32 상수가 256블록마다 붙는다."""
    if blocksize is None:
        return code_bits
    if dq_blocksize2 is None:
        return code_bits + const_bits / blocksize
    return code_bits + dq_const_bits / blocksize + 32 / (blocksize * dq_blocksize2)


def model_gb(n_params, bpp):
    """1GB = 10^9바이트."""
    return n_params * bpp / 8 / 1e9


# ---------- 재기 ----------

def errors(xs, ys):
    n = len(xs)
    diffs = [y - x for x, y in zip(xs, ys)]
    mse = sum(d * d for d in diffs) / n
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / n)
    nz = [(x, y) for x, y in zip(xs, ys) if x != 0]
    return {
        "rmse": math.sqrt(mse),
        "rel_rmse": math.sqrt(mse) / sd if sd else 0.0,
        "max_abs": max(abs(d) for d in diffs),
        "zero_crush": sum(1 for x, y in nz if y == 0) / len(nz) if nz else 0.0,
    }


def synthetic_weights(n=65536, sigma=0.02, seed=6):
    """합성 가중치: 평균 0, 표준편차 sigma인 정규분포에서 n개(정한 값). 실제 모델 가중치가 아니다."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def inject_outliers(xs, every=1000, size=0.5, seed=7):
    """스트레스 시험용: 대략 every개마다 하나를 ±size로 바꾼다(정한 값)."""
    rng = random.Random(seed)
    ys = list(xs)
    idx = sorted(rng.sample(range(len(xs)), len(xs) // every))
    for i in idx:
        ys[i] = size if rng.random() < 0.5 else -size
    return ys, idx


def code_usage(xs, method, blocksize=64):
    """코드(칸)마다 몇 %가 들어갔나. 균일 4bit는 -7..7(15칸), 표 방식은 표 순서대로 16칸."""
    if method == "int4u":
        c, _ = quant_uniform(xs, 7, blocksize)
        keys = list(range(-7, 8))
    else:
        table = {"fp4": fp4_table, "nf4": nf4_table}[method]()
        c, _ = quant_table(xs, table, blocksize)
        keys = list(range(len(table)))
    n = len(c)
    return [sum(1 for v in c if v == k) / n for k in keys]
