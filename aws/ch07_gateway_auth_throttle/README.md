# 7장 · API Gateway·인증·사용량 제한을 어디에 둘까

토큰 버킷(초당 채워지는 토큰과 최대 버킷 크기로 요청을 받는 방식) 제한이 몇 건을 받고 몇 건에 429를 돌려주는지, 클라이언트가 재시도하면 요청이 몇 번으로 불어나는지, 30초짜리 모델이 동기 API 입구를 통과하는지 AWS 없이 계산합니다.

## 실행

```bash
python3 gateway_throttle.py
python3 -m unittest -v
```

## 기대 출력

```
20명 동시, 초당 2개, 버킷 5개
지터 없음 {'attempts': 61, 'served': 20, 'gave_up': 0, 'last_ok_s': 15.0}
지터 {'attempts': 69, 'served': 20, 'gave_up': 0, 'last_ok_s': 11.865}
30초 모델: ['모델 30초가 REST API 통합 제한 시간(29초)에 걸림'] ['모델 30초가 HTTP API 통합 제한 시간(30초)에 걸림']
```

## 검증하는 값 (테스트가 확인)

- 버킷 5, 초당 2개: 동시 10건은 5건 통과, 5건 429. 0.5초마다 1개 회복
- 층이 여러 개면 가장 좁은 층이 결정하고, 거절된 요청은 다른 층의 토큰을 쓰지 않음(단순화)
- 지터 없는 재시도: 20명이 61번 시도하고 마지막 성공은 15.0초. 시드 7의 지터는 69번, 11.865초. 시드 1~8이면 시도 65~70번, 마지막 성공 11.125~20.475초(대기는 0부터 그 차례 상한 1·2·4·8초 사이, 최대 10번 시도)
- 모델 30초는 REST 통합 제한 29초와 HTTP 30초(미만이어야 함)를 모두 못 맞춤

한도 출처(2026-11-14 확인):
<https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html>
<https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-usage-plans.html>
<https://docs.aws.amazon.com/apigateway/latest/developerguide/limits.html>
<https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-execution-service-limits-table.html>
<https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html>

## 이 실습이 확인하지 못하는 것

- 실제 제한은 최선 노력이라 정확한 통과 수를 보장하지 않습니다. 이 모의는 이상적인 버킷입니다.
- 제한 시간 초과 때 클라이언트가 받는 상태 코드는 이 실습이 확인하지 못했습니다(확인 필요).
- 지터 방식(전체 지터)과 최대 시도 횟수는 실습용 선택이지 AWS의 값이 아닙니다.
- 인증 방식(IAM, Cognito, Lambda 권한 부여자)의 동작은 계산하지 않고 본문 표로만 다룹니다.
