# 11장. 간섭과 SUTVA

1000명(커뮤니티 20개 x 50명)의 가상 친구 네트워크에서, 내 쿠폰이 이웃 지출에 번지는 정도(gamma)를 0, +4, -4로 바꿔 가며 개인 단위 무작위 A/B와 커뮤니티 단위 무작위의 추정값을 '전원 처치 대 전원 미처치'의 진짜 효과와 비교합니다.

## 실행

```bash
cd causal/ch11_interference_sutva
python3 interference.py          # 숫자를 출력해요
python3 -m unittest -q  # 테스트 (pytest도 됩니다)
```

파이썬 표준 라이브러리만 쓰고 시드를 고정해서 같은 출력이 나옵니다.

## 기대 출력 (`python3 interference.py`)

```text
mean_degree 11.728
isolated 0
cross_edge_share 0.157
neighbor_share {'individual_treated': 0.5, 'individual_control': 0.5, 'cluster_treated': 0.92, 'cluster_control': 0.08}
no_spillover {'truth': 5.0, 'individual': 4.98, 'cluster': 4.96, 'ind_sd': 0.41, 'clu_sd': 0.35}
spillover_pos {'truth': 9.0, 'individual': 4.97, 'cluster': 8.32, 'ind_sd': 0.41, 'clu_sd': 0.34}
spillover_neg {'truth': 1.0, 'individual': 4.99, 'cluster': 1.61, 'ind_sd': 0.41, 'clu_sd': 0.35}
```

## 이 실습이 확인하지 못하는 것

이웃 비율에 선형으로 번진다는 가장 단순한 간섭 모형만 씁니다. 처치 배정만 100번 다시 뽑고 고객과 잡음은 고정이므로 표준편차는 배정에서 오는 흩어짐입니다. 군집 설계의 분산이나 군집 내 상관, 네트워크를 모를 때의 문제는 다루지 않습니다.
