"""API Gateway의 토큰 버킷 제한과 재시도를 AWS 없이 흉내 내는 계산기.

공식 문서(2026-11-14 확인)가 말하는 구조만 옮겼다.
  https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-usage-plans.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/limits.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-execution-service-limits-table.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html
실제 제한은 "최선 노력"이라 이 모의가 정확한 개수를 보장하지 않는다. 시간은 밀리초 정수로 센다.
"""
import heapq
import random

CHECKED_AT = "2026-11-14"
REST_INTEGRATION_TIMEOUT_S = 29     # 기본 상한. 리전·비공개 API는 올릴 수 있으나 계정 제한 감소가 필요할 수 있음
HTTP_API_INTEGRATION_TIMEOUT_S = 30  # 올릴 수 없음
PAYLOAD_LIMIT_MB = 10


class TokenBucket:
    """rate_per_s: 초당 채워지는 토큰, burst: 버킷 크기. 내부는 1/1000 토큰 단위 정수."""

    def __init__(self, rate_per_s, burst):
        self.rate_mt_per_ms = rate_per_s      # 초당 1토큰 = 1000 밀리토큰/1000ms = 1밀리토큰/ms
        self.capacity = burst * 1000
        self.tokens = self.capacity
        self.t = 0

    def _refill(self, t_ms):
        self.tokens = min(self.capacity, self.tokens + (t_ms - self.t) * self.rate_mt_per_ms)
        self.t = t_ms

    def has_token(self, t_ms):
        self._refill(t_ms)
        return self.tokens >= 1000

    def take(self):
        self.tokens -= 1000

    def allow(self, t_ms):
        if self.has_token(t_ms):
            self.take()
            return True
        return False


def layered_allow(buckets, t_ms):
    """모든 층이 받아야 통과한다. 거절된 요청은 어느 층의 토큰도 쓰지 않는다(단순화)."""
    if all(b.has_token(t_ms) for b in buckets):
        for b in buckets:
            b.take()
        return 200
    return 429


def simulate(arrivals_ms, buckets):
    return [layered_allow(buckets, t) for t in sorted(arrivals_ms)]


class DailyQuota:
    """사용량 계획의 기간 쿼터를 흉내 낸 단순 카운터. 실제 쿼터도 최선 노력이다."""

    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def allow(self):
        if self.used >= self.limit:
            return 429
        self.used += 1
        return 200


def retry_until_served(n_clients, rate_per_s, burst, base_ms=1000, jitter=False, seed=7, max_attempts=10):
    """n명이 동시에 요청하고, 429를 받으면 두 배씩 늘리며 다시 시도한다."""
    bucket = TokenBucket(rate_per_s, burst)
    rng = random.Random(seed)
    heap = [(0, i, 0) for i in range(n_clients)]
    heapq.heapify(heap)
    attempts = served = gave_up = 0
    last_ok_ms = 0
    while heap:
        t, cid, k = heapq.heappop(heap)
        attempts += 1
        if bucket.allow(t):
            served += 1
            last_ok_ms = max(last_ok_ms, t)
        elif k + 1 >= max_attempts:
            gave_up += 1
        else:
            wait = base_ms * (2 ** k)
            if jitter:
                wait = rng.randint(0, wait)  # 전체 지터: 0부터 대기 상한 사이
            heapq.heappush(heap, (t + max(wait, 1), cid, k + 1))
    return {"attempts": attempts, "served": served, "gave_up": gave_up, "last_ok_s": last_ok_ms / 1000}


def average_rps(requests_per_day):
    return requests_per_day / 86400


def capacity_rps(instances, concurrent_per_instance, seconds_per_request):
    """모델 쪽이 계속 받아 낼 수 있는 초당 요청 수의 위쪽 한계."""
    return instances * concurrent_per_instance / seconds_per_request


def sync_path_fits(model_seconds, api_type="rest", payload_mb=0.02):
    """모델 호출 한 번이 동기 API 입구 제한 안에 드는지."""
    limit = REST_INTEGRATION_TIMEOUT_S if api_type == "rest" else HTTP_API_INTEGRATION_TIMEOUT_S
    reasons = []
    if model_seconds >= limit:
        reasons.append(f"모델 {model_seconds}초가 {api_type.upper()} API 통합 제한 시간({limit}초)에 걸림")
    if payload_mb > PAYLOAD_LIMIT_MB:
        reasons.append(f"페이로드 {payload_mb}MB가 {PAYLOAD_LIMIT_MB}MB를 넘음")
    return reasons


if __name__ == "__main__":
    print("20명 동시, 초당 2개, 버킷 5개")
    for jitter in (False, True):
        print("지터" if jitter else "지터 없음", retry_until_served(20, 2, 5, jitter=jitter))
    print("30초 모델:", sync_path_fits(30, "rest"), sync_path_fits(30, "http"))
