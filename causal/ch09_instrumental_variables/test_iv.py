import unittest
import iv


class WaldHand(unittest.TestCase):
    def test_hand_numbers(self):
        rows = iv.simulate()
        self.assertAlmostEqual(iv.itt_y(rows), 4.86, delta=0.02)
        self.assertAlmostEqual(iv.first_stage(rows), 0.503, delta=0.002)
        self.assertAlmostEqual(iv.itt_y(rows) / iv.first_stage(rows), iv.wald(rows))

    def test_wald_close_to_late_not_naive(self):
        rows = iv.simulate()
        self.assertAlmostEqual(iv.wald(rows), 10.0, delta=0.8)
        self.assertGreater(iv.naive(rows), 20)  # 선택 편향으로 크게 틀린다
        self.assertAlmostEqual(iv.true_ate(rows), 5.85, delta=0.1)

    def test_wald_is_not_ate(self):
        rows = iv.simulate()
        self.assertGreater(abs(iv.wald(rows) - iv.true_ate(rows)), 3)


class WeakAndExclusion(unittest.TestCase):
    def test_weak_first_stage_widens_estimates(self):
        strong = iv.repeated(reps=100, n=2000, p_complier=0.5)
        weak = iv.repeated(reps=100, n=2000, p_complier=0.03)
        self.assertLess(strong["p90"] - strong["p10"], 5)
        self.assertGreater(weak["p90"] - weak["p10"], 50)

    def test_first_stage_f(self):
        self.assertGreater(iv.first_stage_f(iv.simulate(n=2000, seed=1000)), 100)
        self.assertLess(iv.first_stage_f(iv.simulate(n=2000, p_complier=0.03, seed=1000)), 10)

    def test_exclusion_violation_amplified_by_weak_stage(self):
        strong = iv.wald(iv.simulate(n=20000, direct=1.0))
        weak = iv.wald(iv.simulate(n=20000, p_complier=0.03, direct=1.0))
        self.assertAlmostEqual(strong, 11.65, delta=0.8)
        self.assertGreater(weak, 25)
        # 약한 도구에서는 F가 10을 넘어도 배제제약 위반의 편향이 크다
        self.assertGreater(iv.first_stage_f(iv.simulate(n=20000, p_complier=0.03)), 10)

    def test_deterministic(self):
        self.assertEqual(iv.wald(iv.simulate()), iv.wald(iv.simulate()))


if __name__ == "__main__":
    unittest.main()
