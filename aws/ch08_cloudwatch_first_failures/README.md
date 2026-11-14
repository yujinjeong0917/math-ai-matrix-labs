# 8장 · CloudWatch에서 어떤 실패를 먼저 봐야 하나

지표 값열(1분마다 한 값)을 경보 규칙으로 따라가며 어떤 경보가 몇 분째에 먼저 울리는지, 누락 데이터 처리 방식이 상태를 어떻게 바꾸는지, 마이크로초·밀리초 단위 착각이 어떻게 항상 울리는 경보를 만드는지 AWS 없이 확인합니다.

## 실행

```bash
python3 alarm_check.py scenario.json
python3 -m unittest -v
```

## 기대 출력

```
backlog_without_capacity: 2   (분 2에 첫 경보)
oldest_age: 3
expired: null  (울리지 않음)
model_latency_vs_gateway: 3
gpu_one_card: 5
errors_5xx_notBreaching: null
errors_5xx_missing: null
wrong_unit_alarm_minutes: 5
```

## 검증하는 값 (테스트가 확인)

- ModelLatency 31,500,000 마이크로초 = 31.5초, 29,000 밀리초 = 29초
- GPUUtilization 200%(GPU 4개) = GPU 1개당 50%
- 모든 점이 누락일 때: notBreaching은 OK, breaching은 ALARM, missing은 INSUFFICIENT_DATA, ignore는 이전 상태 유지
- "용량 0 + 줄 있음" 경보(분 2)가 "가장 오래된 요청 120초 초과" 경보(분 3)보다 1분 빠름

공식 문서(2026-11-14 확인):
<https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-evaluation.html>
<https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarms-and-missing-data.html>
<https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-monitor.html>
<https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html>
<https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-metrics-and-dimensions.html>

## 이 실습이 확인하지 못하는 것

- 실제 CloudWatch는 평가 범위를 N보다 넓게 읽고 성급한 경보를 막는 규칙이 있습니다. 여기서는 최근 N개만 봅니다.
- 시나리오 값은 가상입니다. 실제 엔드포인트가 0대일 때 각 지표가 점을 내는지는 확인 필요.
- 임계값(120초, 80% 등)은 예시이며 서비스 목표에서 정해야 합니다.
- 경보 알림 비용과 지표 보관 비용은 계산하지 않았습니다.
