"""API Gateway의 토큰 버킷 제한과 재시도를 AWS 없이 흉내 내는 계산기.

공식 문서(2026-11-14 확인)가 말하는 구조만 옮겼다.
  https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-usage-plans.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/limits.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-execution-service-limits-table.html
  https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html
실제 제한은 "최선 노력"이라 이 모의가 정확한 개수를 보장하지 않는다. 시간은 밀리초 정수로 센다.

재는 것: 토큰 버킷이 동시 요청 중 몇 건을 받고 몇 건에 429를 돌려주는지, 429 뒤 재시도가
시도 횟수를 얼마나 늘리는지(지터 유무 비교), 30초 모델이 동기 API 입구를 통과하는지.
입력: 코드 안의 가상 값(20명 동시, 초당 2개, 버킷 5개). 출력: README 기대 출력 4줄
(지터 없음 61번 시도 15.0초, 시드 7 지터 69번 11.865초, 30초 모델의 REST 29초/HTTP 30초 탈락).
이상적인 버킷이라 실제 서비스의 최선 노력 제한과 개수가 정확히 같지 않다.
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
        # 정수만 써서 부동소수점 오차를 피한다(1토큰 = 1000밀리토큰, 시간은 ms)
        self.rate_mt_per_ms = rate_per_s      # 초당 1토큰 = 1000 밀리토큰/1000ms = 1밀리토큰/ms
        self.capacity = burst * 1000
        self.tokens = self.capacity
        self.t = 0

    def _refill(self, t_ms):
        # 경과 ms x 채움 속도만큼 더하되 버킷 크기(burst)를 넘지 않는다. t_ms는 단조 증가해야 한다(함정: 되감으면 토큰이 줄어듦)
        self.tokens = min(self.capacity, self.tokens + (t_ms - self.t) * self.rate_mt_per_ms)
        self.t = t_ms

    def has_token(self, t_ms):
        """토큰을 쓰지 않고 1개(1000밀리토큰) 이상 있는지만 본다."""
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
    # 먼저 전부 확인한 뒤에 한꺼번에 차감한다. 그래서 한 층이 거절하면 다른 층 토큰은 그대로(실제 서비스의 동작인지는 확인 필요)
    if all(b.has_token(t_ms) for b in buckets):
        for b in buckets:
            b.take()
        return 200
    return 429


def simulate(arrivals_ms, buckets):
    """도착 시각(ms) 순서대로 처리해 상태 코드 목록(200/429)을 돌려준다."""
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
    rng = random.Random(seed)  # 시드를 고정해 지터가 있어도 실행마다 같은 결과(README 11.865초)
    # 힙 항목: (시각 ms, 클라이언트 번호, 지금까지 실패 횟수). 시각이 같으면 번호순이라 결정적이다
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
            # 지수 백오프: 1, 2, 4, 8초... (실습용 선택, AWS 값 아님)
            wait = base_ms * (2 ** k)
            if jitter:
                wait = rng.randint(0, wait)  # 전체 지터: 0부터 대기 상한 사이
            # 지터로 대기가 0이 되면 같은 ms에 다시 몰리므로 최소 1ms
            heapq.heappush(heap, (t + max(wait, 1), cid, k + 1))
    return {"attempts": attempts, "served": served, "gave_up": gave_up, "last_ok_s": last_ok_ms / 1000}


def average_rps(requests_per_day):
    """하루 요청 수를 초당 평균 요청 수로 바꾼다(20,000건/일 = 0.2315건/초)."""
    return requests_per_day / 86400


def capacity_rps(instances, concurrent_per_instance, seconds_per_request):
    """모델 쪽이 계속 받아 낼 수 있는 초당 요청 수의 위쪽 한계."""
    # 30대 x 1건 / 30초 = 1.0건/초. 피크 3배(0.694)의 1.44배 여유
    return instances * concurrent_per_instance / seconds_per_request


def sync_path_fits(model_seconds, api_type="rest", payload_mb=0.02):
    """모델 호출 한 번이 동기 API 입구 제한 안에 드는지."""
    limit = REST_INTEGRATION_TIMEOUT_S if api_type == "rest" else HTTP_API_INTEGRATION_TIMEOUT_S
    reasons = []
    # "미만"이어야 통과해서 같은 값도 탈락(>=). 그래서 HTTP 30초 모델은 30초 한도에 걸림
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
