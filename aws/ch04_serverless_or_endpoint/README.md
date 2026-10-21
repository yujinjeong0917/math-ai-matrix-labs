# 4장 · 하루 20건인데 서버를 계속 켜 둬야 할까요?

요청 수만으로 AWS 서비스를 고르지 않고, GPU·실행 메모리·건당 시간·허용 대기 시간으로 후보를 먼저 거릅니다. AWS 계정과 비용이 드는 리소스는 만들지 않아요.

## 실행

```bash
python3 deployment_choice.py traffic_profile.json
python3 -m pytest -q test_deployment_choice.py
```

## 먼저 채울 값

`traffic_profile.json`의 `null`은 오류가 아니라 아직 측정하지 않았다는 뜻입니다.

- `region`: 실제로 배포할 AWS 리전
- `runtime_memory_gb`: 모델 파일 크기가 아니라 실행 중 최대 메모리
- `container_image_gb`: 압축을 풀었을 때 컨테이너 이미지 크기
- `gpu_required`: 목표 응답 시간을 CPU로 맞출 수 있는지 재고 입력

하나라도 비어 있으면 결과는 `HOLD`입니다. 후보가 남아도 추천 완료가 아니라 실제 모델을 올려 시작 시간·p95 응답 시간·월 비용을 비교할 후보라는 뜻입니다.

## 검증하는 숫자

- 20건/일 × 30일 = 600건/월
- 600건 × 30초 = 18,000초 = 5시간/월
- 30일 × 24시간 = 720시간/월
- 작업자 1개에 20건이 동시에 오면 1·10·20번째 완료는 30·300·600초
- 시작 시간 90초를 더하면 120·390·690초

## 경계

서비스 한도는 2026-10-22 AWS 공식 문서 스냅샷입니다. 실제 배포 전에 최신 문서와 계정 할당량을 다시 확인하세요. 가격은 리전과 구성에 따라 바뀌므로 코드에 고정하지 않았습니다.

- <https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html>
- <https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html>
- <https://docs.aws.amazon.com/decision-guides/latest/decision-guides/fargate-or-lambda.html>
- <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-tasks-services.html>
- <https://docs.aws.amazon.com/sagemaker/latest/dg/model-deploy-feature-matrix.html>
- <https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html>
- <https://docs.aws.amazon.com/sagemaker/latest/dg/hosting-faqs.html>
- <https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html>
