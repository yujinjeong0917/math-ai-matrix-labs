"""1인 개발 앱 출시 실전 3장: 사용자별 토큰 버킷 사용량 제한.

버킷 하나에 토큰이 최대 B개(용량) 들어가고, 1초에 r개씩(충전 속도) 다시 찬다. 요청 하나가 토큰 하나를 쓴다.
토큰이 1개 이상이면 허용하고 1개를 빼고, 없으면 거부(429 Too Many Requests)하며 다음 토큰까지 남은 초를
Retry-After로 알려 준다. 시간 T 동안 허용되는 요청 수는 B + rT 를 넘지 못한다.

시계는 밖에서 넣어 준다(FakeClock). 그래서 sleep 없이, 매번 같은 결과가 나온다.
소수 오차가 없도록 토큰 수와 시간은 Fraction(분수)으로 센다.
"""

import math
from fractions import Fraction as F

# 용량 B와 충전 속도 r은 실습에서 정한 값이다. 서비스에 맞는 수치는 비용과 사용 패턴으로 따로 정해야 한다.
CAPACITY = 5            # B: 한 번에 몰아 쓸 수 있는 최대 횟수(실습에서 정한 값)
REFILL_PER_SEC = F(1, 12)  # r: 12초에 1개, 1분에 5개(실습에서 정한 값)


class FakeClock:
    def __init__(self, t=0):
        self.t = F(t)

    def __call__(self):
        return self.t

    def set(self, t):
        self.t = F(t)


class TokenBucket:
    def __init__(self, capacity=CAPACITY, rate=REFILL_PER_SEC, now=F(0)):
        self.capacity = F(capacity)
        self.rate = F(rate)
        # 처음에는 가득 찬 상태로 시작한다.
        self.tokens = F(capacity)
        self.last = F(now)

    def _refill(self, now):
        now = F(now)
        # 시계가 거꾸로 가면 충전하지 않는다(음수 충전 방지).
        if now > self.last:
            # 충전량 = 경과 시간 x r, 용량 B를 넘으면 버린다(상한 B).
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now

    def take(self, now):
        """(허용 여부, 거부라면 Retry-After 초)."""
        self._refill(now)
        if self.tokens >= 1:
            self.tokens -= 1
            return True, 0
        # 남은 부족분 (1 - 토큰) / r 초를 올림. 헤더는 정수 초라 올림해야 그 시각에 다시 시도하면 통과한다.
        return False, math.ceil((1 - self.tokens) / self.rate)


class PerUserLimiter:
    """사용자마다 버킷을 따로 둔다."""

    def __init__(self, clock, capacity=CAPACITY, rate=REFILL_PER_SEC):
        self.clock, self.capacity, self.rate = clock, capacity, rate
        self.buckets = {}

    def take(self, user):
        now = self.clock()
        # 새 사용자의 버킷은 처음 본 시각 now에서 가득 찬 채로 시작한다.
        b = self.buckets.setdefault(user, TokenBucket(self.capacity, self.rate, now))
        return b.take(now)


class GlobalCounter:
    """1장 서버처럼 사용자 구분 없이 전체 횟수만 세는 방식(비교용). 한도를 넘으면 모두 거부."""

    def __init__(self, limit):
        self.limit, self.count = limit, 0

    def take(self, user):
        # 거부된 요청도 센다. 한도를 넘으면 이후 모두 거부되고 회복 규칙은 없다(비교용 단순 방식).
        self.count += 1
        return (self.count <= self.limit), 0


class NoLimit:
    def take(self, user):
        return True, 0


def simulate(limiter_take, times, user="alice"):
    """times(초)마다 한 번씩 요청했을 때 허용 수와 첫 거부의 Retry-After."""
    allowed, first_retry, allowed_at = 0, None, []
    for t in times:
        ok, retry = limiter_take(user, t)
        if ok:
            allowed += 1
            allowed_at.append(t)
        elif first_retry is None:
            first_retry = retry
    return {"requests": len(times), "allowed": allowed, "rejected": len(times) - allowed,
            "first_retry_after": first_retry, "allowed_at": allowed_at}


def bucket_take(capacity=CAPACITY, rate=REFILL_PER_SEC):
    buckets = {}

    def take(user, t):
        # 첫 요청 시각 t를 버킷 시작 시각으로 쓴다.
        b = buckets.setdefault(user, TokenBucket(capacity, rate, t))
        return b.take(t)
    return take


def upper_bound(T, capacity=CAPACITY, rate=REFILL_PER_SEC):
    """B + rT (분수)."""
    return F(capacity) + F(rate) * F(T)
