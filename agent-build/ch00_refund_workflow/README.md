# 0장 · 환불 문의 한 건을 사람은 몇 단계로 처리하고, 그중 어디가 되돌릴 수 없나요?

가상 문의 20건(환불 12, 교환 4, 배송조회 4)마다 필요한 도구 호출 순서를 `inquiries.json`에 고정해 두고, 호출 수, 쓰기 호출 수, 되돌릴 수 없는 호출 수를 세요. 가짜 모델도 서버도 없는 단순 집계이고, 파이썬 표준 라이브러리만 쓰며 난수가 없어요.

## 실행

```bash
python3 refund_workflow.py
python3 -m unittest -q
```

## 기대 출력(앞부분)

```
cases 20
by_kind {'환불': 12, '교환': 4, '배송조회': 4}
total_calls 96
total_writes 53
total_irreversible 30
total_money_out 10
naive_mean_calls_all20 4.8
refund_mean_calls 5.083333333333333
```

## 도구 분류(가상)

읽기: `lookup_order`, `get_tracking`, `check_stock`, `get_policy`. 쓰기이면서 되돌릴 수 있음: `create_exchange`(출고 전 취소 가능하다고 가정), `log_note`. 쓰기이면서 되돌릴 수 없음: `issue_refund`, `send_message`.

## 확인하지 못하는 것

- 실제 상담 현장의 단계 수와 걸리는 시간. 문의 20건은 이 장을 위해 만든 값이고 조사한 값이 아니에요.
- 어떤 도구가 되돌릴 수 있는가는 서비스마다 달라요. 환불 취소가 모든 환불에 되는지는 실제 결제 서비스 문서를 열어 확인해야 해요.
- 실제 모델이 문의를 보고 이 호출 순서를 고르는지는 재지 않았어요.
