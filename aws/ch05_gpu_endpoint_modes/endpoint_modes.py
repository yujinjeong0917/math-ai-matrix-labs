"""GPU 엔드포인트 세 방식(실시간·비동기·배치)을 AWS 계정 없이 계산으로 견주는 도구.

AWS를 호출하지 않는다. 한도는 2026-11-14에 연 공식 문서의 값이고,
가격은 어디에도 넣지 않는다(unit_prices는 계산기에서 받아 적는 칸일 뿐).
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

# 공식 문서(feature matrix)의 값. 단위는 MB와 초. "<"는 미만이라는 뜻이라 strict로 따로 적는다.
LIMITS = {
    "realtime": {"payload_mb": 6, "payload_strict": True, "timeout_s": 60, "timeout_strict": True, "gpu": True, "scale_to_zero": False},
    "async": {"payload_mb": 1024, "payload_strict": False, "timeout_s": 3600, "timeout_strict": True, "gpu": True, "scale_to_zero": True},
    "batch": {"payload_mb": 100, "payload_strict": False, "timeout_s": None, "timeout_strict": False, "gpu": True, "scale_to_zero": None},
    "serverless": {"payload_mb": 4, "payload_strict": False, "timeout_s": 60, "timeout_strict": True, "gpu": False, "scale_to_zero": True},
}
BATCH_PAYLOAD_PRODUCT_MB = 100  # MaxConcurrentTransforms * MaxPayloadInMB 는 100MB를 넘지 못한다.


def average_concurrency(requests_per_day, seconds_per_request):
    """리틀의 법칙: 평균 동시 처리 수 = 도착률(건/초) x 건당 처리 시간(초)."""
    return requests_per_day / 86400 * seconds_per_request


def instances_needed(requests_per_day, seconds_per_request, peak_factor=1.0,
                     target_utilization=1.0, concurrent_per_instance=1):
    avg = average_concurrency(requests_per_day, seconds_per_request)
    return math.ceil(round(avg * peak_factor / target_utilization / concurrent_per_instance, 9))


def monthly_instance_hours(instances, days=30):
    return instances * 24 * days


def check_limits(mode, payload_mb, seconds, gpu_required):
    """한도에 걸리는 이유를 문자열 목록으로 돌려준다. 빈 목록이면 통과."""
    lim = LIMITS[mode]
    why = []
    if gpu_required and not lim["gpu"]:
        why.append("GPU 미지원")
    over = payload_mb >= lim["payload_mb"] if lim["payload_strict"] else payload_mb > lim["payload_mb"]
    if over:
        why.append(f"페이로드 {payload_mb}MB가 한도({lim['payload_mb']}MB)를 넘음")
    if lim["timeout_s"] is not None and seconds >= lim["timeout_s"]:
        why.append(f"처리 {seconds}초가 제한 시간({lim['timeout_s']}초 미만)을 못 맞춤")
    return why


def async_burst_completions(n_requests, service_s, startup_s, instances):
    """0대에서 시작해 n건이 동시에 도착했을 때 k번째 완료 시각(초).
    인스턴스는 startup_s에 모두 준비되고 한 번에 1건씩 처리한다고 단순화한다."""
    return [startup_s + math.ceil(k / instances) * service_s for k in range(1, n_requests + 1)]


def desired_instances(backlog, target_per_instance, max_instances):
    """ApproximateBacklogSizePerInstance 목표 추적을 거칠게 흉내 낸다.
    실제 제어기의 쿨다운과 평활은 넣지 않았다."""
    if backlog <= 0:
        return 0
    return min(max_instances, math.ceil(backlog / target_per_instance))


def async_active_seconds_isolated(n_requests, service_s, startup_s, idle_hold_s):
    """요청이 서로 멀리 떨어져 와서 매번 0에서 켜지는 경우의 하루 가동 초."""
    return n_requests * (startup_s + service_s + idle_hold_s)


def async_active_seconds_burst(n_requests, service_s, startup_s, idle_hold_s, instances=1):
    return instances * (startup_s + math.ceil(n_requests / instances) * service_s + idle_hold_s)


def batch_makespan_seconds(n_requests, service_s, input_files, instances):
    """입력 파일 하나는 인스턴스 하나만 처리한다. 일하는 인스턴스 수는 min(파일, 인스턴스)."""
    workers = min(input_files, instances)
    return math.ceil(n_requests / workers) * service_s, workers


def batch_config_ok(max_concurrent_transforms, max_payload_mb):
    return max_payload_mb <= 100 and max_concurrent_transforms * max_payload_mb <= BATCH_PAYLOAD_PRODUCT_MB


def evaluate(profile):
    p = profile
    avg = average_concurrency(p["requests_per_day"], p["seconds_per_request"])
    need = instances_needed(p["requests_per_day"], p["seconds_per_request"], p["peak_factor"],
                            p["target_utilization"], p["concurrent_requests_per_instance"])
    modes = {}
    for mode in ("realtime", "async", "batch", "serverless"):
        modes[mode] = check_limits(mode, p["payload_mb"], p["p95_seconds_per_request"], p["gpu_required"])
    span, workers = batch_makespan_seconds(p["requests_per_day"], p["seconds_per_request"],
                                           p["batch_input_files"], p["batch_instances"])
    return {
        "checked_at": CHECKED_AT,
        "average_concurrency": round(avg, 2),
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
