# 1장 · 문의를 잘못 분류하면 어디까지 번지나요?

<!-- sync:start -->
> [에이전트 만들기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/agent-build) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [문의를 잘못 분류하면 어디까지 번지나요](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/01-ticket-routing.html)

```bash
cd agent-build/ch01_ticket_routing
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `ticket_routing.py` | 1장: 문의 40건(가상)을 키워드 규칙으로 분류하고, 혼동행렬과 오분류 비용(가상)을 센다. |
<!-- sync:end -->

가상 문의 40건(환불 13, 교환 10, 배송조회 9, 기타 8)에 정답 라벨을 붙이고, 키워드 규칙 세 가지(v1~v3)로 분류해 혼동행렬과 오분류 비용(가상 비용표)을 세요. 표준 라이브러리만 쓰고 난수가 없어요.

## 실행

```bash
python3 ticket_routing.py
python3 -m unittest -q
```

## 기대 출력(요약)

| 규칙 | 정답 수 | 사람 확인 | 잘못 실행된 환불 | 비용(가상 원) |
|---|---|---|---|---|
| v1 환불 우선 | 29 | 0 | 8 | 490000 |
| v2 부정 표현 처리 | 31 | 0 | 5 | 313000 |
| v3 환불·교환 같이 나오면 사람 확인 | 28 | 6 | 2 | 142000 |

환불 정밀도/재현율은 v1 11/19, 11/13, v2 10/15, 10/13, v3 9/11, 9/13이에요.

비용표는 `ticket_routing.py`의 `COST`예요. 잘못 실행된 환불은 건당 60000, 환불을 다른 분류로 보낸 비용은 3000~6000, 사람 확인은 건당 1500으로 정한 가상 값이에요.

## 확인하지 못하는 것

- 실제 언어 모델 분류기의 정확도. 이 코드는 키워드 규칙이에요.
- 비용표의 숫자는 가정이에요. 실제 환불 손실, 고객 이탈 비용은 조사하지 않았어요.
- 문의 40건은 이 장을 위해 만든 값이라, 실제 문의 분포와 다를 수 있어요.
