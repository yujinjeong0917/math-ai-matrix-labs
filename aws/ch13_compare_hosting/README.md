# 13장 · AWS 밖의 후보를 조건표로 거르기

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [AWS 밖 서비스에도 같은 조건표를 던지면 무엇이 남나요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/13-compare-hosting.html)

```bash
cd aws/ch13_compare_hosting
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `hosting_filter.py` | AWS 밖 후보(Cloud Run, Azure ML, Cloudflare, 무료 호스팅)를 조건표로 거른다. |
<!-- sync:end -->

Cloud Run, Azure ML, Cloudflare(Workers AI·R2), Vercel, Netlify, Render, Fly.io, Hugging Face Spaces를 같은 입력 조건으로 거릅니다. 특정 서비스를 추천하지 않고 `BLOCKED`, `CANDIDATE`, `CANDIDATE_WITH_UNKNOWNS`로만 나눕니다. 계정·네트워크·비용이 필요 없습니다.

## 실행

```bash
python3 hosting_filter.py profile.json
python3 -m unittest -q
```

## 무엇을 재나

- `candidates.json`: 공식 문서를 2026-11-14에 열어 확인한 값, 출처 URL, 메모. `null`은 확인하지 못한 값입니다.
- 허용 대기(`acceptable_wait_seconds`): 꺼져 있다가 첫 요청이 들어온 뒤 처리를 시작하기까지(시작 지연 + 모델 불러오기) 기다려 줄 수 있는 시간. 건당 30초는 따로 '한 요청 제한 시간'과 비교합니다.
- `profile.json`: 내 서비스의 조건(GPU 필요, 실행 메모리, 건당 시간, 허용 대기, 무료만 쓸지). 모든 값은 예시용 가정입니다.
- 검사: GPU, 내 모델 파일을 올리는 방식, 메모리, 건당 처리 시간, 시작 지연 + 모델 불러오기, 무료 사용 범위, 분당 요청 한도.
- `price_input_sheet`: 가격 대신 가격표에 넣을 양(월 요청 수, 실제 계산 시간, 항상 켜 둔 시간)만 만듭니다. 단가는 `null`입니다.

## 기대 출력(기본 profile)

- 몰리는 구간 분당 요청 0.033건
- Cloud Run `CANDIDATE`, Vercel `BLOCKED`(메모리 4GB < 8GB), 나머지 6곳은 `CANDIDATE_WITH_UNKNOWNS`
- 허용 대기를 300초에서 60초로 줄이면 Cloud Run·Render·Fly.io가 `BLOCKED`
- `free_only`를 켜면 Hugging Face Spaces가 `BLOCKED`
- Cloud Run `CANDIDATE`에는 주의 줄이 붙음: GPU 기본 할당량 0, 메모리 문서 32GiB와 80GiB 불일치(둘 다 8GB보다 큼)
- 모델 파일 4.683GB(파인튜닝 트랙 10장의 7B 양자화 파일)는 4.361GiB라 한 번에 업로드 가능
- 가격표 입력량: 600건, 18,000초, 5시간, 720시간

## 이 실습이 확인하지 못하는 것

- 약관(상업 이용, 콘텐츠 제한). 모든 후보에 직접 확인할 항목으로 남깁니다.
- 가격, 무료 한도의 월 단위 소진 시점, 지역별 제공 여부.
- 실제 모델을 올렸을 때의 시작 시간과 응답 시간. 문서의 시작 시간에는 모델 파일을 불러오는 시간이 들어 있지 않습니다.
- 서비스는 한도를 자주 바꿉니다. 배포 날에 아래 URL을 다시 여세요.

## 출처(2026-11-14 확인)

- https://docs.cloud.google.com/run/quotas
- https://docs.cloud.google.com/run/docs/configuring/services/gpu
- https://docs.cloud.google.com/run/docs/configuring/min-instances
- https://docs.cloud.google.com/run/docs/configuring/request-timeout
- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2
- https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2
- https://developers.cloudflare.com/workers-ai/platform/limits/
- https://developers.cloudflare.com/workers-ai/features/fine-tunes/
- https://developers.cloudflare.com/workers/platform/limits/
- https://developers.cloudflare.com/r2/platform/limits/
- https://vercel.com/docs/limits
- https://vercel.com/docs/plans/hobby
- https://docs.netlify.com/build/functions/background-functions/
- https://render.com/docs/free
- https://docs.fly.io/launch/autostop-autostart
- https://fly.io/blog/wrong-about-gpu/
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-gpus
