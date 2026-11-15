"""9장 테스트: ITT, 첫 단계, Wald의 관계, Wald가 LATE(약 10)에 가깝고 ATE(약 5.8)와 다름, 약한 도구에서 추정이 크게 흔들림,
약한 도구가 배제제약 위반을 증폭함을 확인한다. 허용 오차(delta)는 표본 잡음 크기를 반영한 값이다."""
import unittest
import iv


class WaldHand(unittest.TestCase):
    def test_hand_numbers(self):
        rows = iv.simulate()
        # ITT 약 4.86 = 첫 단계(complier 비율) 0.5 x complier 효과 10. 나누면 Wald가 된다
        self.assertAlmostEqual(iv.itt_y(rows), 4.86, delta=0.02)
        self.assertAlmostEqual(iv.first_stage(rows), 0.503, delta=0.002)
        self.assertAlmostEqual(iv.itt_y(rows) / iv.first_stage(rows), iv.wald(rows))

    def test_wald_close_to_late_not_naive(self):
        rows = iv.simulate()
        self.assertAlmostEqual(iv.wald(rows), 10.0, delta=0.8)
        self.assertGreater(iv.naive(rows), 20)  # 선택 편향으로 크게 틀린다
        # 진짜 ATE는 0.2x4 + 0.5x10 + 0.3x0 = 5.8(표본 비율 때문에 5.85 근처)
        self.assertAlmostEqual(iv.true_ate(rows), 5.85, delta=0.1)

    def test_wald_is_not_ate(self):
        rows = iv.simulate()
        # Wald는 complier의 효과(LATE)라 전체 평균 효과(ATE)와 다르다. 3 넘게 벌어진다
        self.assertGreater(abs(iv.wald(rows) - iv.true_ate(rows)), 3)


class WeakAndExclusion(unittest.TestCase):
    def test_weak_first_stage_widens_estimates(self):
        # 10~90 분위 폭: 강한 도구는 5 미만, 약한 도구(complier 3%)는 50 초과
        strong = iv.repeated(reps=100, n=2000, p_complier=0.5)
        weak = iv.repeated(reps=100, n=2000, p_complier=0.03)
        self.assertLess(strong["p90"] - strong["p10"], 5)
        self.assertGreater(weak["p90"] - weak["p10"], 50)

    def test_first_stage_f(self):
        # 흔히 F > 10을 "약하지 않은 도구"의 관례 기준으로 쓴다. 강한 도구는 100 초과, 약한 도구는 10 미만
        self.assertGreater(iv.first_stage_f(iv.simulate(n=2000, seed=1000)), 100)
        self.assertLess(iv.first_stage_f(iv.simulate(n=2000, p_complier=0.03, seed=1000)), 10)

    def test_exclusion_violation_amplified_by_weak_stage(self):
        strong = iv.wald(iv.simulate(n=20000, direct=1.0))
        weak = iv.wald(iv.simulate(n=20000, p_complier=0.03, direct=1.0))
        # 강한 도구: 10 + 직접 효과 1 / 첫 단계 0.5 = 약 12 근처(11.65)
        self.assertAlmostEqual(strong, 11.65, delta=0.8)
        self.assertGreater(weak, 25)
        # 약한 도구에서는 F가 10을 넘어도 배제제약 위반의 편향이 크다
        self.assertGreater(iv.first_stage_f(iv.simulate(n=20000, p_complier=0.03)), 10)

    def test_deterministic(self):
        # 시드 고정이라 같은 설정을 두 번 돌리면 정확히 같은 값
        self.assertEqual(iv.wald(iv.simulate()), iv.wald(iv.simulate()))


if __name__ == "__main__":
    unittest.main()
