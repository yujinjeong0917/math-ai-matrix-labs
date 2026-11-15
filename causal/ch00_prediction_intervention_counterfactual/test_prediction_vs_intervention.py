"""0장 테스트: 관찰 격차는 진짜 효과를 크게 과대평가하고, 무작위 배정 격차는 진짜 효과 근처로 돌아오는지 확인한다.
기대값(2.29, 0.27, 15.01 등)은 고정 시드로 한 번 돌려 얻은 값이다. 같은 시드면 항상 같다."""
import unittest

import prediction_vs_intervention as p


class SceneA(unittest.TestCase):
    def test_observed_gap_is_large_but_randomized_gap_is_small(self):
        a = p.scene_a()
        # 진짜 효과 0.3에 비해 관찰 격차는 약 7.6배: 열성도 h가 섞인 결과
        self.assertEqual(round(a["observed_gap"], 2), 2.29)
        self.assertEqual(round(a["randomized_gap"], 2), 0.27)
        self.assertGreater(a["observed_gap"], 5 * a["true_effect"])
        self.assertAlmostEqual(a["randomized_gap"], a["true_effect"], delta=0.1)  # 표본 잡음 허용폭 0.1

    def test_deterministic(self):
        # 시드 고정이라 두 번 호출한 dict가 완전히 같아야 한다(부동소수점도 비트 단위로 동일)
        self.assertEqual(p.scene_a(), p.scene_a())


class SceneB(unittest.TestCase):
    def test_naive_before_after_swallows_the_season(self):
        b = p.scene_b(0.0)
        self.assertEqual(round(b["naive"], 2), 15.01)
        self.assertEqual(round(b["season_rise"], 2), 10.56)
        # season_rise(10.56)가 단순 전후(15.01)의 대부분을 차지한다
        self.assertGreater(b["naive"], 2 * b["true_effect"])

    def test_seasonal_counterfactual_recovers_effect_when_only_season_moves(self):
        b = p.scene_b(0.0)
        self.assertEqual(round(b["seasonal_cf"], 2), 4.46)
        m = b["means"]
        self.assertEqual([round(m[k], 2) for k in ("this_pre", "this_post", "last_pre", "last_post")],
                         [112.83, 127.84, 102.43, 112.97])
        self.assertAlmostEqual(b["seasonal_cf"], b["true_effect"], delta=1.5)  # 노이즈 sd 2가 평균에 남는 만큼 허용

    def test_simultaneous_event_breaks_the_seasonal_counterfactual(self):
        b = p.scene_b(12.0)
        self.assertEqual(round(b["naive"], 2), 27.01)
        self.assertEqual(round(b["seasonal_cf"], 2), 16.46)
        # 광고 증액 +12가 효과로 들어와 어림값이 진짜 5보다 10 넘게 크다(반사실 가정이 깨진 경우)
        self.assertGreater(b["seasonal_cf"] - b["true_effect"], 10)


if __name__ == "__main__":
    unittest.main()
