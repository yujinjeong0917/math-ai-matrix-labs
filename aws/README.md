# AWS 배포·운영

[저장소 전체 목차](https://github.com/yujinjeong0917/math-ai-matrix-labs) · [웹에서 이 트랙 보기](https://math-ai-matrix-pi.vercel.app/tracks/aws/00-local-to-service.html) · 장 15개

## 실행

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q aws   # 1/15장만 `uv` 환경 필요
```

## 장 목록

| 폴더 | 웹 챕터 제목 | 웹 |
|---|---|---|
| [`ch00_local_to_service/`](ch00_local_to_service/) | 내 노트북에서 잘 돌던 모델이 서비스가 되려면 무엇이 더 필요할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/00-local-to-service.html) |
| [`ch01_request_path/`](ch01_request_path/) | 요청 한 건은 어디를 지나 답이 될까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/01-request-path.html) |
| [`ch02_artifact_versions/`](ch02_artifact_versions/) | 모델 파일과 실행 환경은 어디에 두고 어떻게 버전을 맞출까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/02-artifacts-s3-ecr.html) |
| [`ch03_iam_least_privilege/`](ch03_iam_least_privilege/) | 누가 무엇을 할 수 있어야 할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/03-iam-least-privilege.html) |
| [`ch04_serverless_or_endpoint/`](ch04_serverless_or_endpoint/) | 하루 20건인데 서버를 계속 켜 둬야 할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/04-serverless-or-endpoint.html) |
| [`ch05_gpu_endpoint_modes/`](ch05_gpu_endpoint_modes/) | GPU 엔드포인트는 언제 켜 둘 가치가 있나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/05-gpu-endpoint-modes.html) |
| [`ch06_bedrock_vs_self_hosting/`](ch06_bedrock_vs_self_hosting/) | Bedrock과 직접 호스팅은 무엇을 서로 대신하지 못하나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/06-bedrock-vs-self-hosting.html) |
| [`ch07_gateway_auth_throttle/`](ch07_gateway_auth_throttle/) | API Gateway·인증·사용량 제한을 어디에 둘까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/07-api-gateway-auth-throttle.html) |
| [`ch08_cloudwatch_first_failures/`](ch08_cloudwatch_first_failures/) | CloudWatch에서 어떤 실패를 먼저 봐야 하나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/08-cloudwatch-first-failures.html) |
| [`ch09_canary_rollback/`](ch09_canary_rollback/) | 새 모델은 몇 건을 보고 나서 전체로 바꿔야 할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/09-canary-rollback.html) |
| [`ch10_cost_leaks/`](ch10_cost_leaks/) | 청구서는 요청 수가 아니라 어디에서 커질까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/10-cost-leaks.html) |
| [`ch11_retry_duplicates/`](ch11_retry_duplicates/) | 재시도가 같은 문의를 두 번 처리하면 어떻게 하나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/11-retry-duplicates.html) |
| [`ch12_prompt_log_privacy/`](ch12_prompt_log_privacy/) | 고객이 쓴 문장을 로그에 얼마나 남겨도 될까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/12-prompt-log-privacy.html) |
| [`ch13_compare_hosting/`](ch13_compare_hosting/) | AWS 밖 서비스에도 같은 조건표를 던지면 무엇이 남나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/13-compare-hosting.html) |
| [`ch14_small_to_scaled_architecture/`](ch14_small_to_scaled_architecture/) | 하루 20건의 구성을 2만 건까지 그대로 키우면 무엇이 먼저 막히나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/aws/14-small-to-scaled-architecture.html) |
