# 1장 · 요청 한 건은 어디를 지나가나

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [요청 한 건은 어디를 지나 답이 될까요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/01-request-path.html)

```bash
cd aws/ch01_request_path
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `request_path.py` | 요청 한 건의 경로를 단계 표로 만들고, 가장 작은 시간 한도·크기 한도에 걸리는지 검사한다. |
<!-- sync:end -->

요청 한 건의 경로를 단계 표로 만들고, 경로에서 가장 작은 시간 한도·크기 한도에 걸리는지 검사합니다. 한도 값은 `limits_2026-11-14.json`에 공식 문서 URL과 함께 적어 두었어요. 단계별 시간(ms)은 가상 값입니다. AWS 계정과 네트워크는 쓰지 않아요.

## 실행

```bash
python3 request_path.py
python3 -m unittest -q
```

## 기대 출력(일부)

```text
[30초 모델] REST API + Lambda: 탈락 / 모델 최대 28.75초 / ['API 입구 뒤 구간 30.25초 > apigw_rest_integration_timeout_s 29초']
[30초 모델] HTTP API + Lambda: 탈락 / 모델 최대 29.75초 / ['API 입구 뒤 구간 30.25초 > apigw_http_integration_timeout_s 30초']
[REST 28.8초] 탈락 ['API 입구 뒤 구간 29.05초 > apigw_rest_integration_timeout_s 29초']
[7MiB 입력] 탈락 ['입력 7340032바이트 > lambda_sync_payload_mb 6291456바이트']
```

## 확인하지 못하는 것

- DNS·TLS·Lambda 시작 시간은 가상 값입니다(Lambda 시작 200ms + 로그 50ms = 0.25초가 API 입구 뒤 구간의 덧붙는 시간). 실제 값은 직접 재야 해요.
- 한도는 2026-11-14 문서 기준입니다. REST API의 통합 제한 시간은 Regional·private API에서 할당량 증가를 요청할 수 있다고 적혀 있지만 얼마까지 늘어나는지는 확인하지 못했고, 이 실습은 29초로만 계산해요.
- API Gateway에서 Lambda 없이 SageMaker 엔드포인트를 바로 부르는 구성은 확인하지 않았습니다.
- 가격은 다루지 않아요.
