# 10장 · 에이전트가 맞게 처리했는지 어떻게 재야 하나요?

<!-- sync:start -->
> [에이전트 만들기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/agent-build) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [에이전트가 맞게 처리했는지 어떻게 재야 하나요?](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/10-agent-evaluation.html)

```bash
cd agent-build/ch10_agent_evaluation
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `evaluate.py` | 10장: 기록(trace) 20개를 결과만 채점할 때와 과정까지 채점할 때 비교한다. 전부 가상 기록이에요. |
<!-- sync:end -->

가상 문의 20건의 처리 기록(trace) 20개를 두 방식으로 채점한다. 결과 채점은 환불 서버를 다시 돌려서 실제로 나간 돈이 정답 금액과 같은지만 본다. 과정 채점은 기록의 도구 호출 순서를 규칙 세 개로 본다(환불 전에 주문 조회, 가상 문턱 50 이상 금액은 사람 승인 먼저, 같은 주문에 키가 다른 환불 두 번 금지). 기록은 `evaluate.py` 안의 표로 만들어지며 난수는 없다. 표준 라이브러리만 쓴다.

## 실행

```bash
python3 evaluate.py              # 결과를 JSON으로 출력
python3 -m unittest -v           # 테스트 8개
```

## 기대 출력(핵심)

| 채점 | 통과(20건 중) | 95% 신뢰구간(윌슨) |
|---|---|---|
| v1 결과만 | 18 | 0.699 ~ 0.972 |
| v1 과정만 | 14 | |
| v1 결과와 과정 모두 | 13 | 0.433 ~ 0.819 |
| v2(규칙 수정 후) 결과와 과정 모두 | 15 | 0.531 ~ 0.888 |

- v1에서 결과는 맞는데 과정이 틀린 기록은 5건(승인 건너뜀 3건, 조회 없이 환불 2건)이다.
- v2는 승인 건너뜀 3건을 고쳤지만 4번 문의가 금액을 5 더 보낸다. 통과 수는 13에서 15로 올랐는데 같은 세트에서 v1에는 통과하고 v2에는 실패한 문의가 1건(4번) 있다.

## 이 실습이 확인하지 못하는 것

기록은 규칙으로 만든 가상 기록이라 실제 에이전트가 문제를 얼마나 자주 일으키는지는 모른다. 20건은 신뢰구간이 넓다는 것을 보이려는 표본이지 권장 크기가 아니다. 모델이 매긴 점수(LLM-as-judge)와 사람 점수의 일치도는 재지 않는다. 같은 문의를 여러 번 돌렸을 때의 일관성(pass^k)도 재지 않는다.

## 출처(2026-11-14 확인)

- <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>
- <https://arxiv.org/abs/2406.12045>
