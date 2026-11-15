# 5장 · GPU 엔드포인트는 언제 켜 둘 가치가 있나

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [GPU 엔드포인트는 언제 켜 둘 가치가 있나요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/05-gpu-endpoint-modes.html)

```bash
cd aws/ch05_gpu_endpoint_modes
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `endpoint_modes.py` | GPU 엔드포인트 세 방식(실시간·비동기·배치)을 AWS 계정 없이 계산으로 견주는 도구. |
<!-- sync:end -->

SageMaker 실시간·비동기·배치 세 방식을 "몇 대가 필요한가", "한도에 걸리는가", "0대에서 깨어나면 몇 초 늦는가"로 견주는 계산 실습입니다. AWS 계정이 없어도 돌고, AWS를 호출하지 않으며, 가격은 어디에도 넣지 않았어요.

## 실행

```bash
python3 endpoint_modes.py endpoint_profile.json
python3 -m unittest -v
```

## 먼저 채울 값

`endpoint_profile.json`의 숫자는 모두 가상입니다. 내 모델에서 잰 값으로 바꿔 쓰세요. `region`과 `unit_prices`는 비워 둔 칸이며, 단가는 AWS Pricing Calculator에서 배포할 리전 기준으로 받아 적습니다.

## 기대 출력 (같은 JSON일 때)

- 평균 동시 처리 수 6.94 (20,000건/일 x 30초 / 86,400초)
- 피크 3배, 목표 가동률 0.7이면 인스턴스 30대, 한 달 상시 가동 21,600 인스턴스·시간
- 서버리스는 `GPU 미지원`으로 탈락, 실시간·비동기·배치는 한도 통과
- 입력 파일 1개·인스턴스 4대이면 배치는 한 대만 일해 166.67시간

## 검증하는 값 (테스트가 확인)

- 하루 20건 x 30초 = 600초, 20건이 동시에 오면 0대에서 시작한 비동기의 1·10·20번째 완료는 120·390·690초
- 인스턴스 4대로 늘리면 20번째 완료가 240초
- 백로그 20건, 목표 5건/인스턴스이면 희망 인스턴스 4대 (거친 근사)
- 한도: 실시간 6MB 미만·60초 미만, 비동기 1GB 이하·1시간 미만, 배치 100MB 이하, 서버리스 GPU 없음

한도 출처(2026-11-14 확인):
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-deploy-feature-matrix.html>
<https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html>
<https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html>
<https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html>

## 이 실습이 확인하지 못하는 것

- 실제 모델의 건당 시간, 모델 적재 시간, GPU 메모리. 직접 재야 합니다.
- 실제 자동 확장 제어기의 쿨다운과 평활. 여기서는 `올림(백로그 / 목표)`로 거칠게 근사했습니다.
- `idle_hold_seconds`(마지막 요청 뒤 켜 두는 시간) 300초는 공식 문서의 축소 쿨다운 예시값(ScaleInCooldown 300)을 빌린 가상 값입니다. 실제 기본값은 확인 필요.
- 한 인스턴스가 동시에 몇 건을 처리하는지는 모델·서빙 컨테이너마다 다릅니다(`concurrent_requests_per_instance`).
- 금액. 단가를 입력해도 청구액이 아니라 추정입니다.
