# 7장 · 고객 메시지나 주문 메모 안의 지시를 에이전트가 따르면 어떻게 되나요?

<!-- sync:start -->
> [에이전트 만들기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/agent-build) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [고객 메시지나 주문 메모 안의 지시를 에이전트가 따르면 어떻게 되나요?](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/07-prompt-injection.html)

```bash
cd agent-build/ch07_prompt_injection
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `prompt_injection.py` | 7장 · 고객 메시지나 주문 메모 안의 "지시"를 에이전트가 따르면 어떻게 되나요? (방어·점검용 장난감) |
<!-- sync:end -->

방어·점검용 장난감이에요. 실제 서비스나 실제 모델을 건드리지 않고, 규칙표로 움직이는 가짜 에이전트만 써요. 가짜 에이전트는 "글에 지시처럼 들리는 말(`AGENT_CUES`)이 있고 글이 요구하는 동작이 있으면 따른다"는 약한 규칙 하나로 고정돼 있어요. 입력 20건(고객 메시지 8, 주문 메모 6, 택배 기록 6)을 방어 변형 5개에 넣고 무단 실행 건수를 세요. 표준 라이브러리만 쓰고 난수가 없어요.

## 실행

```bash
python3 prompt_injection.py
python3 -m unittest -v             # 시험 7개
```

## 기대 출력

| 변형 | 무단 실행 | 도구가 거절 | 자료로만 읽어 따르지 않음 | 정당한 환불 완료(15건) | 평범한 글이 빠진 수 |
|---|---|---|---|---|---|
| V0 방어 없음 | 14 | 0 | 0 | 15 | 0 |
| V1 키워드 걸러내기 | 11 | 0 | 0 | 13 | 2 |
| V2 데이터와 지시 분리 | 6 | 0 | 8 | 15 | 0 |
| V3 도구 쪽 검사 | 3 | 11 | 0 | 15 | 0 |
| V4 V2 + V3 | 1 | 5 | 8 | 15 | 0 |

남는 건은 허용된 도구(환불 사유 기록)를 허용된 주문에 쓰는 동작이에요. 도구 쪽 검사가 "값이 맞는가"는 모른다는 뜻이에요.

## 이 실습이 확인하지 못하는 것

실제 모델이 이런 글을 따를지, 이 가짜 규칙이 실제 모델의 약점을 얼마나 대표하는지, 우회 표현 전반, 글의 출처 구분이 섞이는 경우, 최종 답변 문장으로 정보가 새는 경로는 재지 않았어요. 걸러내기가 몇 건을 막는지는 어구 목록을 어떻게 정했느냐에 따라 정해지는 값이고, 이 실습은 공격 문장을 수집하거나 최적화하지 않아요.

## 출처(2026-11-14 확인)

- <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>
- <https://genai.owasp.org/llmrisk/llm062025-excessive-agency/>
- <https://openai.github.io/openai-agents-python/guardrails/>
