# 3장 · 조회할 때와 환불할 때 사이에 주문 상태가 바뀌면 어떻게 되나요?

에이전트의 단계(조회 → 판단 → 실행)와 다른 작업자의 단계를 줄 세우는 모든 순서를 `itertools.combinations`로 나열하고, 순서마다 한 번씩 차례로 실행해 낡은 조회로 실행된 건수, 돈이 두 번 나간 건수, 전이표가 막은 건수를 세요. 표준 라이브러리만 쓰고 난수와 시간이 없어요.

## 실행

```bash
python3 stale_order_state.py
python3 -m unittest -q
```

## 기대 출력(합계 줄)

```
V0 합계 {'orders': 38, 'attempted': 22, 'stale_attempts': 18, 'executed': 22, 'blocked': 0, 'double_payout': 6, 'unsafe_refund': 18}
V1 합계 {'orders': 38, 'attempted': 22, 'stale_attempts': 18, 'executed': 4, 'blocked': 18, 'double_payout': 0, 'unsafe_refund': 0}
V2 합계 {'orders': 60, 'attempted': 11, 'stale_attempts': 7, 'executed': 11, 'blocked': 0, 'double_payout': 2, 'unsafe_refund': 7}
```

V0는 조회 결과만 믿고 실행, V1은 실행할 때 전이표로 다시 검사, V2는 실행 직전에 한 번 더 조회만 해요.

## 확인하지 못하는 것

- 실제 스레드 경쟁과 데이터베이스 격리 수준. 이 코드는 순서를 줄 세워 하나씩 실행해요. 한 단계 안에서 일어나는 겹침은 재지 않아요.
- V1의 "실행할 때 검사"가 실제 시스템에서 원자적으로(검사와 변경이 쪼개지지 않게) 되려면 저장소가 그 기능을 지원해야 해요. 이 코드는 검사와 변경을 한 걸음에 처리한다고 가정해요.
- 4장의 멱등성 키가 이 틈을 막는지는 재지 않았어요. 두 작업자는 서로 다른 경로에서 요청을 보내므로 같은 키를 쓰지 않아요.
- 작업자와 전이표는 이 장을 위해 만든 가상 값이에요.
