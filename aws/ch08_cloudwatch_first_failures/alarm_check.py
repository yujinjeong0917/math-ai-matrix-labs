"""CloudWatch 경보가 지표 값열을 보고 언제 울리는지 AWS 없이 따라가 보는 검사기.

공식 문서(2026-11-14 확인)의 평가 설정 세 가지(Period, Evaluation Periods, Datapoints to Alarm)와
누락 데이터 처리 네 가지만 옮겼다.
  https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-evaluation.html
  https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarms-and-missing-data.html
  https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-monitor.html
  https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html
단순화한 것: 실제 CloudWatch는 평가 범위를 N보다 넓게 읽고 "성급한 경보 방지" 규칙을 쓴다.
여기서는 가장 최근 N개만 본다.
"""
import json
import sys

OK, ALARM, INSUFFICIENT = "OK", "ALARM", "INSUFFICIENT_DATA"
MISSING_MODES = ("missing", "notBreaching", "breaching", "ignore")


def us_to_s(microseconds):
    """SageMaker ModelLatency의 단위는 마이크로초다."""
    return microseconds / 1_000_000


def ms_to_s(milliseconds):
    """API Gateway Latency·IntegrationLatency의 단위는 밀리초다."""
    return milliseconds / 1000


def normalize_gpu(raw_percent, gpu_count):
    """GPUUtilization은 0~100에 GPU 수를 곱한 범위다. 한 장 기준으로 나누려면 GPU 수로 나눈다."""
    return raw_percent / gpu_count


def evaluate_alarm(values, threshold, evaluation_periods=1, datapoints_to_alarm=1,
                   treat_missing="missing", previous_state=OK, greater=True):
    """values: 오래된 것부터 최근 것까지, 없으면 None. 가장 최근 evaluation_periods개를 본다."""
    if treat_missing not in MISSING_MODES:
        raise ValueError(treat_missing)
    window = values[-evaluation_periods:]
    if all(v is None for v in window):
        if treat_missing == "missing":
            return INSUFFICIENT
        if treat_missing == "ignore":
            return previous_state
    breaching = 0
    for v in window:
        if v is None:
            if treat_missing == "breaching":
                breaching += 1
        elif (v > threshold) if greater else (v < threshold):
            breaching += 1
    return ALARM if breaching >= datapoints_to_alarm else OK


def replay(values, threshold, **kw):
    """매 분 평가했을 때의 상태 열."""
    states, prev = [], OK
    for i in range(1, len(values) + 1):
        prev = evaluate_alarm(values[:i], threshold, previous_state=prev, **kw)
        states.append(prev)
    return states


def first_index(states, state=ALARM):
    return states.index(state) if state in states else None


def run(scenario):
    s = scenario["series"]
    wait = scenario["allowed_wait_seconds"]
    gpus = scenario["gpu_count"]
    lat_s = [None if v is None else us_to_s(v) for v in s["ModelLatency_us"]]
    gpu_one = [normalize_gpu(v, gpus) for v in s["GPUUtilization_raw"]]
    results = {
        "backlog_without_capacity": replay(s["HasBacklogWithoutCapacity"], 0.5, evaluation_periods=2, datapoints_to_alarm=2),
        "oldest_age": replay(s["ApproximateAgeOfOldestRequest_s"], wait),
        "expired": replay(s["ExpiredRequests"], 0),
        "model_latency_vs_gateway": replay(lat_s, scenario["gateway_integration_timeout_seconds"]),
        "gpu_one_card": replay(gpu_one, 80),
        "errors_5xx_notBreaching": replay(s["Invocation5XXErrors"], 0, treat_missing="notBreaching"),
        "errors_5xx_missing": replay(s["Invocation5XXErrors"], 0, treat_missing="missing"),
    }
    # 단위를 헷갈려 29,000(밀리초 감각)을 마이크로초 지표에 그대로 걸었을 때
    wrong = replay(s["ModelLatency_us"], 29000)
    summary = {k: first_index(v) for k, v in results.items()}
    summary["wrong_unit_alarm_minutes"] = sum(1 for x in wrong if x == ALARM)
    return {"states": results, "first_alarm_minute": summary}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scenario.json"
    with open(path, encoding="utf-8") as f:
        out = run(json.load(f))
    print(json.dumps(out["first_alarm_minute"], ensure_ascii=False, indent=2))
