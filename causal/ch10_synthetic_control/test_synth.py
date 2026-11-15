"""10장 테스트: 가중치가 합 1·비음수 조건을 지키는지, 볼록껍질 안에서 효과 약 10을 복원하고 가짜 처치 검정에서 1위인지,
균등 가중치(단순 평균 대조)가 덜 정확한지, 볼록껍질 밖에서는 처치 전부터 못 맞춰 효과가 크게 틀리는지 확인한다.
숫자 기준은 report()가 만드는 값이며, 허용 오차는 잡음(표준편차 0.6)을 반영한 값이다."""
import unittest
import synth as s


class Fit(unittest.TestCase):
    def test_weights_on_simplex(self):
        tr, dn = s.make_panel(in_hull=True)
        w = s.fit_weights(tr, dn)
        self.assertAlmostEqual(sum(w), 1.0, places=9)  # 좌표 이동이 합을 보존한다
        self.assertTrue(all(x >= 0 for x in w))

    def test_good_fit_recovers_effect(self):
        a = s.report()["in_hull"]
        self.assertLess(a["pre_rmspe"], 1.0)  # 처치 전 맞춤 오차가 잡음(0.6) 수준
        self.assertAlmostEqual(a["effect"], 10.0, delta=0.6)  # 진짜 효과 10 복원
        self.assertEqual(a["placebo_rank"], 1)  # 가짜 처치 8개보다 큰 후/전 비율

    def test_equal_weights_less_accurate(self):
        # 도너 8개 단순 평균은 추세가 다른 도너까지 섞여 합성 가중치보다 오차가 크다
        a = s.report()["in_hull"]
        self.assertGreater(abs(a["equal_weight_did"] - 10), abs(a["effect"] - 10))

    def test_pre_fit_failure_gives_wrong_answer(self):
        a = s.report()["out_hull"]
        self.assertGreater(a["pre_rmspe"], 10)  # 처치 전부터 못 맞춘다
        self.assertGreater(a["treated_pre_mean"], a["max_donor_pre_mean"])  # 도너 어느 것보다 높다
        self.assertGreater(abs(a["effect"] - 10), 50)  # 효과 추정이 크게 틀린다(진짜 효과 10 대비 오차 50 초과)
        self.assertGreater(a["placebo_rank"], 1)  # 가짜 처치 검정으로도 두드러지지 않는다

    def test_deterministic(self):
        # 시드 고정이라 같은 값이 나와야 한다(좌표 탐색에 난수가 없다)
        self.assertEqual(s.report()["in_hull"]["effect"], s.report()["in_hull"]["effect"])


if __name__ == "__main__":
    unittest.main()
