# 에이전트 만들기

[저장소 전체 목차](https://github.com/yujinjeong0917/math-ai-matrix-labs) · [웹에서 이 트랙 보기](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/00-refund-workflow.html) · 장 12개

## 실행

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q agent-build   # 1/12장만 `uv` 환경 필요
```

## 장 목록

| 폴더 | 웹 챕터 제목 | 웹 |
|---|---|---|
| [`ch00_refund_workflow/`](ch00_refund_workflow/) | 환불 문의 한 건의 도구 호출 순서에서 어디부터 되돌릴 수 없나요 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/00-refund-workflow.html) |
| [`ch01_ticket_routing/`](ch01_ticket_routing/) | 문의를 잘못 분류하면 어디까지 번지나요 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/01-ticket-routing.html) |
| [`ch02_tool_arguments/`](ch02_tool_arguments/) | 도구에 넘기는 값이 형식은 맞는데 뜻이 틀리면 누가 잡나요 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/02-tool-arguments.html) |
| [`ch03_stale_order_state/`](ch03_stale_order_state/) | 조회할 때와 환불할 때 사이에 주문 상태가 바뀌면 어떻게 되나요 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/03-stale-order-state.html) |
| [`ch04_double_refund/`](ch04_double_refund/) | 환불이 두 번 나갔어요 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/04-double-refund.html) |
| [`ch05_approval_threshold/`](ch05_approval_threshold/) | 금액이 얼마부터 사람에게 물어야 하나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/05-approval-threshold.html) |
| [`ch06_scoped_permissions/`](ch06_scoped_permissions/) | 에이전트가 이 고객의 주문만 만지게 하려면 어디서 막아야 하나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/06-scoped-permissions.html) |
| [`ch07_prompt_injection/`](ch07_prompt_injection/) | 고객 메시지나 주문 메모 안의 지시를 에이전트가 따르면 어떻게 되나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/07-prompt-injection.html) |
| [`ch08_context_limit/`](ch08_context_limit/) | 대화와 도구 결과가 쌓이면 에이전트는 환불 정책의 어느 조항을 잃나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/08-context-limit.html) |
| [`ch09_cost_and_latency/`](ch09_cost_and_latency/) | 문의 한 건을 처리하는 데 호출이 몇 번이고 사용자는 얼마나 기다리나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/09-cost-and-latency.html) |
| [`ch10_agent_evaluation/`](ch10_agent_evaluation/) | 에이전트가 맞게 처리했는지 어떻게 재야 하나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/10-agent-evaluation.html) |
| [`ch11_trace_and_recovery/`](ch11_trace_and_recovery/) | 틀린 환불이 나갔을 때 원인을 어떻게 찾고 어떻게 바로잡나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/11-trace-and-recovery.html) |
