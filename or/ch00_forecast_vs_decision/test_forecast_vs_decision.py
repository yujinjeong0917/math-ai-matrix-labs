import unittest
from fractions import Fraction
import forecast_vs_decision as f


class TestCh0(unittest.TestCase):
    def test_mae(self):
        # P는 C에서만 8 틀리고(8/3), Q는 A에서만 10 틀린다(10/3): 지역 3곳 평균
        self.assertEqual(f.mae(f.FORECAST_P), Fraction(8, 3))
        self.assertEqual(f.mae(f.FORECAST_Q), Fraction(10, 3))

    def test_better_forecast_worse_decision(self):
        # P: C 38 + B 22 중 실제로 팔리는 건 C 30, B 22 = 270,000 + 110,000 = 380,000
        # Q: B 30 + C 30 = 150,000 + 270,000 = 420,000
        pp = f.profit(f.allocate(f.FORECAST_P))
        pq = f.profit(f.allocate(f.FORECAST_Q))
        self.assertEqual(pp, 380000)
        self.assertEqual(pq, 420000)
        self.assertLess(f.mae(f.FORECAST_P), f.mae(f.FORECAST_Q))
        self.assertLess(pp, pq)

    def test_oracle_matches_brute_force(self):
        self.assertEqual(f.profit(f.allocate(f.ACTUAL)), f.brute_force_best())
        self.assertEqual(f.brute_force_best(), 420000)

    def test_same_forecast_different_stock(self):
        # 재고만 늘려도(60 -> 90) 같은 예측에서 배분이 바뀐다: 결정은 예측과 제약의 함수
        self.assertEqual(f.allocate(f.FORECAST_P, 60), {"A": 0, "B": 22, "C": 38})
        self.assertEqual(f.allocate(f.FORECAST_P, 90), {"A": 22, "B": 30, "C": 38})


if __name__ == "__main__":
    unittest.main()
