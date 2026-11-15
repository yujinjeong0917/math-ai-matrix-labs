"""GPU 엔드포인트 세 방식(실시간·비동기·배치)을 AWS 계정 없이 계산으로 견주는 도구.

AWS를 호출하지 않는다. 한도는 2026-11-14에 연 공식 문서의 값이고,
가격은 어디에도 넣지 않는다(unit_prices는 계산기에서 받아 적는 칸일 뿐).

재는 것: (1) 평균 동시 처리 수와 피크 대비 필요 인스턴스 수, (2) 방식별 한도 위반, (3) 비동기가
0대에서 깨어날 때의 완료 시각, (4) 배치 소요 시간.
입력: endpoint_profile.json(가상 숫자). 출력: evaluate()의 JSON.
README 기대 출력 대응: 평균 동시 6.94, 인스턴스 30대, 21,600 인스턴스·시간,
서버리스 GPU 미지원, 입력 파일 1개면 배치 166.67시간.
"""
import json
import math
import sys

CHECKED_AT = "2026-11-14"
SOURCES = {
    "feature_matrix": "https://docs.aws.amazon.com/sagemaker/latest/dg/model-deploy-feature-matrix.html",
    "async": "https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html",
    "async_autoscale": "https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html",
    "batch": "https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html",
}

# payload_strict/timeout_strict가 True면 "미만"만 통과(같으면 탈락), False면 "이하"까지 통과.
# 공식 문서(feature matrix)의 값. 단위는 MB와 초. "<"는 미만이라는 뜻이라 strict로 따로 적는다.
LIMITS = {
    "realtime": {"payload_mb": 6, "payload_strict": True, "timeout_s": 60, "timeout_strict": True, "gpu": True, "scale_to_zero": False},
    "async": {"payload_mb": 1024, "payload_strict": False, "timeout_s": 3600, "timeout_strict": True, "gpu": True, "scale_to_zero": True},
    "batch": {"payload_mb": 100, "payload_strict": False, "timeout_s": None, "timeout_strict": False, "gpu": True, "scale_to_zero": None},
    "serverless": {"payload_mb": 4, "payload_strict": False, "timeout_s": 60, "timeout_strict": True, "gpu": False, "scale_to_zero": True},
}
# 배치 타임아웃은 이 표에 없어 None(검사 안 함), scale_to_zero None은 "모름"
BATCH_PAYLOAD_PRODUCT_MB = 100  # MaxConcurrentTransforms * MaxPayloadInMB 는 100MB를 넘지 못한다.


def average_concurrency(requests_per_day, seconds_per_request):
    """리틀의 법칙: 평균 동시 처리 수 = 도착률(건/초) x 건당 처리 시간(초)."""
    return requests_per_day / 86400 * seconds_per_request


def instances_needed(requests_per_day, seconds_per_request, peak_factor=1.0,
                     target_utilization=1.0, concurrent_per_instance=1):
    """피크를 감안해 필요한 인스턴스 수(대). 평균 동시 수 x 피크 배수 / 목표 가동률 / 인스턴스당 동시 처리 수를 올림."""
    avg = average_concurrency(requests_per_day, seconds_per_request)
    # 함정: 6.0000000001 같은 부동소수점 잔차가 올림으로 한 대를 더 만들지 않도록 9자리 반올림 후 올림
    # 6.94 -> 7대, 피크 3배 / 가동률 0.7 -> 29.76 -> 30대, 인스턴스당 4건이면 8대
    return math.ceil(round(avg * peak_factor / target_utilization / concurrent_per_instance, 9))


def monthly_instance_hours(instances, days=30):
    """상시 가동 시 월 인스턴스·시간. 30대 x 24시간 x 30일 = 21,600."""
    return instances * 24 * days


def check_limits(mode, payload_mb, seconds, gpu_required):
    """한도에 걸리는 이유를 문자열 목록으로 돌려준다. 빈 목록이면 통과."""
    lim = LIMITS[mode]
    why = []
    if gpu_required and not lim["gpu"]:
        why.append("GPU 미지원")
    # 실시간은 6MB "미만", 비동기는 1GB "이하"처럼 경계 포함 여부가 방식마다 달라 strict로 가른다
    over = payload_mb >= lim["payload_mb"] if lim["payload_strict"] else payload_mb > lim["payload_mb"]
    if over:
        why.append(f"페이로드 {payload_mb}MB가 한도({lim['payload_mb']}MB)를 넘음")
    # 제한 시간은 모두 "미만" 조건이라 같으면 탈락(>=)
    if lim["timeout_s"] is not None and seconds >= lim["timeout_s"]:
        why.append(f"처리 {seconds}초가 제한 시간({lim['timeout_s']}초 미만)을 못 맞춤")
    return why


def async_burst_completions(n_requests, service_s, startup_s, instances):
    """0대에서 시작해 n건이 동시에 도착했을 때 k번째 완료 시각(초).
    인스턴스는 startup_s에 모두 준비되고 한 번에 1건씩 처리한다고 단순화한다."""
    # 시작 시간은 한 번만, 이후 k번째 요청은 ceil(k/대수)번째 차례에 끝난다
    # 1대: 120, 390, 690초(1, 10, 20번째). 4대: 120, 180, 240초
    return [startup_s + math.ceil(k / instances) * service_s for k in range(1, n_requests + 1)]


def desired_instances(backlog, target_per_instance, max_instances):
    """ApproximateBacklogSizePerInstance 목표 추적을 거칠게 흉내 낸다.
    실제 제어기의 쿨다운과 평활은 넣지 않았다."""
    if backlog <= 0:
        return 0  # 비동기는 0대까지 줄 수 있다(scale_to_zero)
    return min(max_instances, math.ceil(backlog / target_per_instance))


def async_active_seconds_isolated(n_requests, service_s, startup_s, idle_hold_s):
    """요청이 서로 멀리 떨어져 와서 매번 0에서 켜지는 경우의 하루 가동 초."""
    # 20건 x (90 + 30 + 300) = 8,400초 = 2.33시간/일. 켜 두는 시간(idle_hold)이 가동 시간을 지배한다
    return n_requests * (startup_s + service_s + idle_hold_s)


def async_active_seconds_burst(n_requests, service_s, startup_s, idle_hold_s, instances=1):
    """한꺼번에 몰려 한 번만 켜지는 경우의 하루 가동 초. 1대: 90 + 20 x 30 + 300 = 990초."""
    return instances * (startup_s + math.ceil(n_requests / instances) * service_s + idle_hold_s)


def batch_makespan_seconds(n_requests, service_s, input_files, instances):
    """입력 파일 하나는 인스턴스 하나만 처리한다. 일하는 인스턴스 수는 min(파일, 인스턴스)."""
    # 함정: 입력 파일이 1개면 인스턴스를 늘려도 일하는 건 1대(README: 1파일 4대 -> 600,000초 = 166.67시간)
    workers = min(input_files, instances)
    return math.ceil(n_requests / workers) * service_s, workers


def batch_config_ok(max_concurrent_transforms, max_payload_mb):
    """동시 변환 수 x 건당 페이로드가 100MB를 넘지 않는지(4 x 25 = 100은 통과, 5 x 25는 탈락)."""
    return max_payload_mb <= 100 and max_concurrent_transforms * max_payload_mb <= BATCH_PAYLOAD_PRODUCT_MB


def evaluate(profile):
    """프로필 하나로 위 계산을 모아 JSON용 사전으로 만든다."""
    p = profile
    avg = average_concurrency(p["requests_per_day"], p["seconds_per_request"])
    need = instances_needed(p["requests_per_day"], p["seconds_per_request"], p["peak_factor"],
                            p["target_utilization"], p["concurrent_requests_per_instance"])
    modes = {}
    for mode in ("realtime", "async", "batch", "serverless"):
        # 한도는 최악 지연(p95)으로 검사하고, 인스턴스 수는 평균 처리 시간으로 계산한다
        modes[mode] = check_limits(mode, p["payload_mb"], p["p95_seconds_per_request"], p["gpu_required"])
    span, workers = batch_makespan_seconds(p["requests_per_day"], p["seconds_per_request"],
                                           p["batch_input_files"], p["batch_instances"])
    return {
        "checked_at": CHECKED_AT,
        "average_concurrency": round(avg, 2),  # 6.94
        "instances_for_peak": need,
        "always_on_instance_hours_month": monthly_instance_hours(need),
        "limit_failures": modes,
        "batch_makespan_hours": round(span / 3600, 2),
        "batch_workers": workers,
        "price_note": "단가는 입력하지 않았다. unit_prices가 null이면 금액을 계산하지 않는다.",
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "endpoint_profile.json"
    with open(path, encoding="utf-8") as f:
        print(json.dumps(evaluate(json.load(f)), ensure_ascii=False, indent=2))
