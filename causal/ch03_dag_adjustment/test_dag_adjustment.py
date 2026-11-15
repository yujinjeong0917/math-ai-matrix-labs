"""3장 테스트: 손계산 표의 정확한 값, 자체 OLS 구현의 정확성, 통제 조합별 계수 표(관찰·무작위), 시드 흔들림 크기를 확인한다.
표의 숫자는 시드 31 고정 결과를 소수 둘째 자리로 반올림한 값이다."""
import unittest

import dag_adjustment as d


class Hand(unittest.TestCase):
    def test_stratified_hand_numbers(self):
        h = d.hand_stratified()
        # 받은 사람 (600x32 + 100x14)/700 = 29.4286, 못 받은 사람 (100x30 + 200x12)/300 = 18.0
        self.assertAlmostEqual(h["treated_mean"], 29.4286, places=4)
        self.assertEqual(h["control_mean"], 18.0)
        self.assertAlmostEqual(h["naive"], 11.4286, places=4)
        self.assertEqual(h["within"], {"높음": 2.0, "낮음": 2.0})
        self.assertEqual(h["weights"], {"높음": 0.7, "낮음": 0.3})
        self.assertAlmostEqual(h["adjusted"], 2.0, places=9)


class Ols(unittest.TestCase):
    def test_ols_recovers_exact_line(self):
        # 잡음 없는 직선이라 정규방정식 해가 절편 3, 기울기 2와 정확히 같아야 한다(소수 9자리)
        rows = [{"x": float(i), "y": 3.0 + 2.0 * i} for i in range(10)]
        b = d.ols(rows, "y", ["x"])
        self.assertAlmostEqual(b["1"], 3.0, places=9)
        self.assertAlmostEqual(b["x"], 2.0, places=9)


class Adjustment(unittest.TestCase):
    def test_observational_table(self):
        # 아무것도 안 넣으면 교란으로 5.00(진짜 2.0), Z만 넣으면 1.99, 매개 M을 넣으면 직접 효과 쪽(0.5)으로 줄어든다
        got = {k: round(v, 2) for k, v in d.adjustment_table()}
        self.assertEqual(got, {
            "아무것도 안 넣음": 5.00, "충성도 Z": 1.99, "Z + 장바구니 M": 0.52,
            "Z + 후기 점수 C": -0.51, "전부 다(Z, M, C)": -0.72})

    def test_only_confounder_adjustment_recovers_total_effect(self):
        got = dict(d.adjustment_table())
        self.assertAlmostEqual(got["충성도 Z"], d.TRUE_TOTAL, delta=0.1)  # 표본 잡음 허용폭 0.1
        self.assertGreater(got["아무것도 안 넣음"] - d.TRUE_TOTAL, 2.5)   # 교란을 안 넣으면 크게 틀린다
        self.assertLess(got["Z + 장바구니 M"], 1.0)                       # 매개를 넣으면 전체 효과가 사라진다
        self.assertLess(got["Z + 후기 점수 C"], 0.0)                      # 충돌변수를 넣으면 부호가 뒤집힌다
        self.assertLess(got["전부 다(Z, M, C)"], 0.0)

    def test_randomized_table_shows_bad_controls_still_hurt(self):
        # 동전 배정이면 교란이 없어 "아무것도 안 넣음"도 약 2.0이지만, 매개와 충돌변수를 넣으면 여전히 망가진다
        got = {k: round(v, 2) for k, v in d.adjustment_table(randomize=True)}
        self.assertEqual(got, {
            "아무것도 안 넣음": 1.96, "충성도 Z": 1.99, "Z + 장바구니 M": 0.44,
            "Z + 후기 점수 C": -0.51, "전부 다(Z, M, C)": -0.72})


class Spread(unittest.TestCase):
    def test_seed_spread_explains_small_gaps(self):
        # 12개 시드 평균으로 표본 흔들림을 확인한다. 계수 차이가 작을 때 시드 탓인지 구조 탓인지 가르는 용도
        sp = d.seed_spread()
        m, s = sp[(True, "아무것도 안 넣음")]
        self.assertAlmostEqual(m, 2.0, delta=0.1)
        self.assertLess(s, 0.1)
        m, s = sp[(True, "Z + 장바구니 M")]
        self.assertAlmostEqual(m, d.DIRECT, delta=0.1)
        m, s = sp[(False, "Z + 장바구니 M")]
        self.assertAlmostEqual(m, d.DIRECT, delta=0.1)


if __name__ == "__main__":
    unittest.main()
