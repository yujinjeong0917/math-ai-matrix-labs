import unittest

from attention_check import d_from_F, EXP1, EXP2, EXP2_MEDIATION, EXP1_N, EXP2_N, f_from_summary, d_from_summary, eta_squared_from_F, mediation_check


class Checks(unittest.TestCase):
    def test_hand_d(self):
        # 손계산: 평균 차이 0.90, SD 제곱 평균 (1.0404+1.1881)/2=1.11425, 제곱근 1.0556 -> 0.85
        self.assertAlmostEqual(d_from_summary(*EXP1["static"], *EXP1["dynamic"]), 0.853, places=3)
        self.assertAlmostEqual(d_from_summary(*EXP2["static"], *EXP2["dynamic"]), 0.970, places=3)

    def test_eta(self):
        self.assertAlmostEqual(eta_squared_from_F(24.38, 1, 196), 0.1106, places=4)
        self.assertAlmostEqual(eta_squared_from_F(35.88, 1, 409), 0.0807, places=4)

    def test_d_from_reported_F_differs_from_d_from_means(self):
        # 보고 F에서 거꾸로 구한 d는 평균·SD에서 구한 d(0.85, 0.97)와 다르다.
        self.assertAlmostEqual(d_from_F(EXP1["F"], EXP1["df2"]), 0.705, places=3)
        self.assertAlmostEqual(d_from_F(EXP2["F"], EXP2["df2"]), 0.592, places=3)
        self.assertGreater(d_from_F(EXP1["F"], EXP1["df2"]), d_from_F(EXP2["F"], EXP2["df2"]))  # 방향이 평균·SD 쪽과 반대

    def test_f_from_summary_differs_from_reported(self):
        r1 = f_from_summary(*EXP1["dynamic"], EXP1_N[0], *EXP1["static"], EXP1_N[1])
        r2 = f_from_summary(*EXP2["dynamic"], EXP2_N[0], *EXP2["static"], EXP2_N[1])
        self.assertAlmostEqual(r1["F"], 35.98, places=2)
        self.assertAlmostEqual(r2["F"], 96.61, places=2)
        self.assertAlmostEqual(r1["eta2"], 0.155, places=3)
        self.assertAlmostEqual(r2["eta2"], 0.191, places=3)
        self.assertGreater(r1["F"], EXP1["F"] * 1.4)  # 논문 보고 F 24.38, 35.88과 크게 다르다
        self.assertGreater(r2["F"], EXP2["F"] * 2.5)

    def test_mediation_product_matches_reported(self):
        m = EXP2_MEDIATION
        r = mediation_check(m["a"], m["b"], m["c_prime"], -1.15)
        self.assertAlmostEqual(r["ab"], -0.5904, places=4)
        self.assertAlmostEqual(round(r["ab"], 2), m["ab_reported"])

    def test_total_does_not_match_mean_difference(self):
        m = EXP2_MEDIATION
        r = mediation_check(m["a"], m["b"], m["c_prime"], -1.15)
        self.assertAlmostEqual(r["total"], -0.6804, places=4)
        self.assertGreater(abs(r["gap"]), 0.4)  # 논문 안에서 맞지 않는 곳


if __name__ == "__main__":
    unittest.main()
