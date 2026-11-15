# 6장 · Bedrock과 직접 호스팅은 무엇을 서로 대신하지 못하나

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [Bedrock과 직접 호스팅은 무엇을 서로 대신하지 못하나요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/06-bedrock-vs-self-hosting.html)

```bash
cd aws/ch06_bedrock_vs_self_hosting
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `hosting_fit.py` | Bedrock Custom Model Import(직접 가져오기)가 내 모델 조건을 받아 주는지 가리는 검사기와, |
<!-- sync:end -->

내 모델을 Bedrock Custom Model Import(가져온 모델을 Bedrock 입구로 서비스하는 기능)가 받아 주는지 조건별로 검사하고, 토큰 쿼터 차감과 5분 청구 창을 계산합니다. AWS를 호출하지 않고 가격을 넣지 않았어요.

## 실행

```bash
python3 hosting_fit.py model_profile.json
python3 -m unittest -v
```

## 기대 출력 (같은 JSON일 때)

`cmi_possible: false`와 막는 이유 3개: 가중치가 GGUF(Hugging Face .safetensors가 아님), 리전 ap-northeast-2가 지원 목록에 없음, Qwen2.5 구조는 Converse API 미지원.

## 검증하는 숫자 (테스트가 확인)

- 시작 시점 차감 = 입력 + max_tokens: 800 + 4,096 = 4,896, 800 + 512 = 1,312
- 가상 쿼터 분당 100,000토큰이면 시작할 수 있는 동시 요청이 20건 대 76건
- 완료 후 정산 = 입력 + 출력 x 소각률: 1,000 + 100 x 5 = 1,500 (문서 예시와 같음)
- 하루 20건이 1시간 간격이면 청구 창 20개, 복사본 1개·CMU 4(가상)이면 400 CMU·분

공식 문서(2026-11-14 확인):
<https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html>
<https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html>
<https://docs.aws.amazon.com/bedrock/latest/userguide/import-model-calculate-cost.html>
<https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html>

## 이 실습이 확인하지 못하는 것

- 쿼터 100,000은 설명용 가상 값입니다. Custom Model Import에 적용되는 쿼터 값은 이 실습이 열어 본 문서에 없어 확인 필요.
- 모델이 유휴 뒤 얼마 만에 내려가는지(그래서 청구 창이 몇 개 더 생기는지)는 문서에서 찾지 못해 `billing_windows`는 하한 추정입니다.
- 지원 구조·리전·형식은 바뀔 수 있으니 배포 날 문서를 다시 여세요.
- "128K 미만"의 K가 1,000인지 1,024인지 문서에 없어 경계 구간은 경고만 합니다.
