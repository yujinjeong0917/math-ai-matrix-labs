# 8장 · 대화와 도구 결과가 쌓이면 에이전트는 환불 정책의 어느 조항을 잃나요?

<!-- sync:start -->
> [에이전트 만들기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/agent-build) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [대화와 도구 결과가 쌓이면 에이전트는 환불 정책의 어느 조항을 잃나요?](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/08-context-limit.html)

```bash
cd agent-build/ch08_context_limit
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `context_limit.py` | 8장 · 대화와 도구 결과가 쌓이면 에이전트는 환불 정책의 어느 조항을 잃나요? (표준 라이브러리만) |
<!-- sync:end -->

컨텍스트에 글자 수 예산(6,000자, 가상)을 두고, 정책 8조항과 대화·도구 결과를 쌓다가 예산을 넘으면 어떤 글이 먼저 잘려 나가는지 세요. 토큰 대신 글자 수를 써요. 정책 길이는 코드의 문장 글자 수이고 턴 길이는 턴 번호로 정한 가상 값이에요. 가짜 에이전트는 조항이 있으면 올바르게 처리하고 없으면 환불을 승인해요. 표준 라이브러리만 쓰고 난수가 없어요.

## 실행

```bash
python3 context_limit.py
python3 -m unittest -v             # 시험 9개
```

## 기대 출력(일부)

- 머리말 46자, 정책 8조항 126·108·130·110·104·95·112·102자(합계 887자), 예산을 처음 넘는 턴은 7이에요.
- 6턴 5,493자, 7턴 6,268자(268자 초과).

| 방식 | 7턴 | 8턴 | 12턴 | 30턴 |
|---|---|---|---|---|
| S1 앞에서부터 자르기: 잃은 조항 | P1, P2 | P1~P8 | P1~P8 | P1~P8 |
| S2 정책 고정: 잘린 고객 메시지 | 1 | 2 | 6 | 24 |
| S3 정책 고정 + 도구 결과 지우기: 잘린 고객 메시지 | 0 | 0 | 0 | 17 |

## 이 실습이 확인하지 못하는 것

실제 토큰 수, 긴 글에서 모델이 조항을 놓치는 정도(context rot), 요약의 품질, 실제 서비스가 컨텍스트를 자르는 방식은 재지 않았어요. 모델별 창 크기는 바뀌므로 고정하지 않았어요.

## 출처(2026-11-14 확인)

- <https://platform.claude.com/docs/en/build-with-claude/context-windows>
