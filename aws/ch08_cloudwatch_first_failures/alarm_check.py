"""CloudWatch 경보가 지표 값열을 보고 언제 울리는지 AWS 없이 따라가 보는 검사기.

공식 문서(2026-11-14 확인)의 평가 설정 세 가지(Period, Evaluation Periods, Datapoints to Alarm)와
누락 데이터 처리 네 가지만 옮겼다.
  https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-evaluation.html
  https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarms-and-missing-data.html
  https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-monitor.html
  https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html
단순화한 것: 실제 CloudWatch는 평가 범위를 N보다 넓게 읽고 "성급한 경보 방지" 규칙을 쓴다.
여기서는 가장 최근 N개만 본다.

재는 것: 지표 값열(1분마다 한 값)에서 경보가 몇 분째에 처음 울리는지, 누락 데이터 처리 방식에 따른 상태,
단위 착각(마이크로초 지표에 밀리초 임계값)이 만드는 오경보.
입력: scenario.json(가상 지표 시계열). 출력: 지표별 첫 경보 분(없으면 null).
README 기대 출력 대응: backlog 2, oldest_age 3, model_latency 3, gpu_one_card 5, 나머지 null,
wrong_unit_alarm_minutes 5. 분은 0부터 센다.
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
    # 함정: 값이 evaluation_periods개보다 적으면 있는 것만 본다(처음 몇 분)
    window = values[-evaluation_periods:]
    if all(v is None for v in window):
        # 창 전체가 누락일 때만 처리 방식이 상태를 결정한다. notBreaching/breaching은 아래 계산으로 이어짐
        if treat_missing == "missing":
            return INSUFFICIENT
        if treat_missing == "ignore":
            return previous_state
    breaching = 0
    for v in window:
        if v is None:
            if treat_missing == "breaching":
                breaching += 1
        # 임계값과 같은 값은 위반이 아니다(초과/미만만 위반)
        elif (v > threshold) if greater else (v < threshold):
            breaching += 1
    # M-of-N: 창 N개 중 위반이 M개 이상이면 ALARM
    return ALARM if breaching >= datapoints_to_alarm else OK


def replay(values, threshold, **kw):
    """매 분 평가했을 때의 상태 열."""
    # 매 분 그 분까지의 값만 보고 평가한다. ignore가 쓰는 이전 상태는 직전 분의 결과
    states, prev = [], OK
    for i in range(1, len(values) + 1):
        prev = evaluate_alarm(values[:i], threshold, previous_state=prev, **kw)
        states.append(prev)
    return states


def first_index(states, state=ALARM):
    """상태 열에서 처음 그 상태가 된 분(0부터). 없으면 None."""
    return states.index(state) if state in states else None


def run(scenario):
    """시나리오의 지표 여러 개를 각각 알맞은 임계값과 단위로 재생한다. 임계값은 예시."""
    s = scenario["series"]
    wait = scenario["allowed_wait_seconds"]
    gpus = scenario["gpu_count"]
    lat_s = [None if v is None else us_to_s(v) for v in s["ModelLatency_us"]]
    gpu_one = [normalize_gpu(v, gpus) for v in s["GPUUtilization_raw"]]
    results = {
        # 선행 지표: 2분 연속(2-of-2)이라 분 1이 아니라 분 2에 울림. 1-of-1이면 한 분 빠름(테스트 참고)
        "backlog_without_capacity": replay(s["HasBacklogWithoutCapacity"], 0.5, evaluation_periods=2, datapoints_to_alarm=2),
        # 가장 오래된 요청의 나이(초)가 허용 대기 시간을 넘으면. 후행 지표라 분 3
        "oldest_age": replay(s["ApproximateAgeOfOldestRequest_s"], wait),
        "expired": replay(s["ExpiredRequests"], 0),
        # 단위를 초로 맞춘 뒤 API 입구 제한 시간과 비교
        "model_latency_vs_gateway": replay(lat_s, scenario["gateway_integration_timeout_seconds"]),
        # GPU 수로 나눈 한 장 기준 80%
        "gpu_one_card": replay(gpu_one, 80),
        # 같은 값열이라도 누락 처리에 따라 상태 열이 OK 또는 INSUFFICIENT_DATA로 갈린다(첫 ALARM은 둘 다 없음)
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
