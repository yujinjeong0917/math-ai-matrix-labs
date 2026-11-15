"""AI 도구 실전 2장: 프롬프트 캐시 모형과 비용 계산.

- 가격표 읽기(data/prices_*.json)
- 손익분기 호출 수 n*: 처음 한 번 쓰기(w배) + 나머지 읽기(r배)가 캐싱 없이 n번 보내는 값(n배)보다 싸지는 가장 작은 n
- 앞부분(프리픽스) 캐시 흉내: 블록 단위로 "처음부터 똑같은 앞부분"만 다시 쓰고, 유지 시간(TTL)이 지나면 지운다.
  읽으면 유지 시간이 다시 시작된다(Anthropic 문서의 refresh). 최소 토큰 수보다 짧으면 캐시하지 않는다.
- usage 두 모양: Anthropic(input_tokens에 캐시 토큰이 빠짐), OpenAI(input_tokens에 캐시 토큰이 들어 있음)

표준 라이브러리만 쓴다. 금액은 Fraction으로 계산해 반올림 오차가 없다.
"""

import json
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
PRICE_FILE = HERE / "data" / "prices_2026-10-10.json"
MTOK = 1_000_000


def frac(x):
    """JSON 숫자를 문자열 경유로 Fraction 변환한다(이진 부동소수점 근사를 피함)."""
    return None if x is None else F(str(x))


def load_prices(path=PRICE_FILE):
    """가격표 JSON을 읽는다. 단가 단위는 달러 / 100만 토큰."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def price_of(model, prices=None):
    """모델 id로 가격표 항목을 찾는다. 없으면 KeyError."""
    prices = prices or load_prices()
    for m in prices["models"]:
        if m["model"] == model:
            return m
    raise KeyError(model)


def multipliers(m, ttl="5m"):
    """기본 입력가를 1로 둔 쓰기 배율 w, 읽기 배율 r. 쓰기 할증이 없는 제공사는 w = 1."""
    base = frac(m["input"])
    w_price = m["write_5m"] if ttl == "5m" else m["write_1h"]
    w = F(1) if w_price is None else frac(w_price) / base
    r = frac(m["read"]) / base
    return w, r


# ---------- 손익분기 ----------

def breakeven_calls(w, r):
    """같은 앞부분을 n번 보낼 때 캐싱(쓰기 1번 + 읽기 n-1번)이 캐싱 없음보다 싸지는 가장 작은 n.

    w + (n-1) r < n  <=>  n > (w - r) / (1 - r).  (r < 1 가정)
    """
    w, r = F(w), F(r)
    bound = (w - r) / (1 - r)
    # 부등식이 엄격(<)이라 bound가 정수일 때 n=bound는 같아질 뿐 이득이 아니다. int(bound)+1이 항상 그보다 큰 가장 작은 정수(bound>0일 때).
    n = int(bound) + 1          # bound보다 큰 가장 작은 정수
    return max(n, 1)


def breakeven_bruteforce(w, r, limit=1000):
    """검증용: 공식 대신 n을 1부터 직접 늘려 가며 첫 이득 지점을 찾는다(O(limit)). 공식 결과와 대조한다."""
    w, r = F(w), F(r)
    for n in range(1, limit):
        if w + (n - 1) * r < n:
            return n
    return None


def prefix_cost_ratio(n, w, r):
    """앞부분 비용만 보았을 때 캐싱 / 캐싱 없음 = (w + (n-1) r) / n."""
    return (F(w) + (n - 1) * F(r)) / n


# ---------- 캐시 흉내 ----------

class PrefixCache:
    """블록 단위 앞부분 캐시.

    요청 = 블록 목록 [(key, tokens), ...] + 캐시 지점(breakpoints, 블록 번호 목록).
    - 찾기: 마지막 캐시 지점 이하의 블록 경계 가운데, 예전에 써 둔 앞부분과 처음부터 똑같은 가장 긴 것을 찾는다.
      (실제 Anthropic은 캐시 지점마다 20블록까지만 거슬러 찾는다. 이 장의 요청은 블록이 4개라 그 한계에 닿지 않는다.)
    - 쓰기: 캐시 지점에서만 쓴다. 찾은 곳 뒤부터 마지막 캐시 지점까지가 쓰기 토큰이다.
    - 마지막 캐시 지점 뒤는 캐시와 상관없는 일반 입력이다.
    - 읽거나 쓰면 그 앞부분의 만료 시각을 지금 + TTL로 다시 잡는다.
    - 마지막 캐시 지점까지의 토큰이 min_tokens보다 적으면 캐시하지 않는다(모두 일반 입력).
    """

    def __init__(self, ttl_minutes, min_tokens):
        self.ttl = ttl_minutes
        self.min_tokens = min_tokens
        # 키를 튜플로 두는 이유: 처음부터 끝까지 같은 블록 열일 때만 같은 키가 되어 '앞부분 일치'가 된다.
        self.store = {}  # 앞부분(블록 key 튜플) -> 만료 시각(분)

    def request(self, blocks, breakpoints, t):
        keys = [k for k, _ in blocks]
        toks = [n for _, n in blocks]
        total = sum(toks)
        bps = sorted(breakpoints)
        # 캐시 지점이 없으면 전부 일반 입력.
        if not bps:
            return {"input_tokens": total, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        last = bps[-1]
        # 마지막 캐시 지점까지의 토큰 수. 최소 토큰 수 판정과 쓰기량 계산의 기준.
        upto = sum(toks[: last + 1])
        if upto < self.min_tokens:
            return {"input_tokens": total, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        matched = -1
        # 긴 앞부분부터 짧은 쪽으로 찾아 가장 긴 일치를 잡는다. 만료(exp < t)된 항목은 없는 것으로 본다.
        for i in range(last, -1, -1):
            key = tuple(keys[: i + 1])
            exp = self.store.get(key)
            if exp is not None and exp >= t:
                matched = i
                self.store[key] = t + self.ttl      # 읽으면 유지 시간이 다시 시작
                break
        read = sum(toks[: matched + 1])
        # 읽고 남은, 마지막 캐시 지점까지의 구간이 새로 쓰는 토큰.
        write = upto - read
        for b in bps:
            # 이미 읽은 지점 이하는 위에서 만료만 갱신했고, 그보다 뒤 지점만 새로 쓴다.
            if b > matched:
                self.store[tuple(keys[: b + 1])] = t + self.ttl
        # Anthropic usage 모양: input_tokens는 캐시 지점 뒤의 일반 입력만이다(캐시 토큰 제외).
        return {"input_tokens": total - upto, "cache_creation_input_tokens": write, "cache_read_input_tokens": read}


# ---------- usage -> 금액 ----------

def cost_anthropic_usage(u, m, ttl="5m", output_tokens=0):
    """Anthropic usage: 전체 입력 = cache_read + cache_creation + input_tokens (input_tokens에 캐시 토큰이 없다)."""
    write_price = m["write_5m"] if ttl == "5m" else m["write_1h"]
    write_price = m["input"] if write_price is None else write_price
    c = (u["input_tokens"] * frac(m["input"])
         + u["cache_creation_input_tokens"] * frac(write_price)
         + u["cache_read_input_tokens"] * frac(m["read"])
         + output_tokens * frac(m["output"]))
    return c / MTOK


def to_openai_usage(u):
    """같은 요청을 OpenAI 모양으로: input_tokens가 전체이고 그 안에 cached_tokens·cache_write_tokens가 들어 있다."""
    total = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
    return {"input_tokens": total,
            "input_tokens_details": {"cached_tokens": u["cache_read_input_tokens"],
                                     "cache_write_tokens": u["cache_creation_input_tokens"]}}


def cost_openai_usage(u, m, output_tokens=0):
    """OpenAI 모양 usage를 금액(달러)으로 바꾼다."""
    d = u["input_tokens_details"]
    # OpenAI 모양은 input_tokens가 전체라서 캐시 토큰을 빼야 일반 입력이 남는다. 안 빼면 이중 계산이 된다.
    plain = u["input_tokens"] - d["cached_tokens"] - d["cache_write_tokens"]
    write_price = m["input"] if m["write_5m"] is None else m["write_5m"]
    c = (plain * frac(m["input"]) + d["cache_write_tokens"] * frac(write_price)
         + d["cached_tokens"] * frac(m["read"]) + output_tokens * frac(m["output"]))
    return c / MTOK


def cost_openai_double_count(u, m):
    """흔한 실수: input_tokens 전체를 기본가로 내고, cached_tokens를 읽기 값으로 또 더한다."""
    d = u["input_tokens_details"]
    return (u["input_tokens"] * frac(m["input"]) + d["cached_tokens"] * frac(m["read"])) / MTOK


def cost_nocache(total_input, m, output_tokens=0):
    """캐싱이 없을 때의 비용(달러). 비교 기준."""
    return (total_input * frac(m["input"]) + output_tokens * frac(m["output"])) / MTOK


# ---------- 시나리오 실행 ----------

def run_calls(requests, ttl_minutes, min_tokens):
    """requests: [(blocks, breakpoints, t)], 결과: usage 목록."""
    # 호출 묶음 하나마다 새 캐시로 시작한다(앞선 실행의 캐시가 남지 않음).
    cache = PrefixCache(ttl_minutes, min_tokens)
    return [cache.request(b, bp, t) for b, bp, t in requests]


def sum_usage(usages):
    out = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    for u in usages:
        for k in out:
            out[k] += u[k]
    return out
