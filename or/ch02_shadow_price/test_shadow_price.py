import unittest
from fractions import Fraction as F
import shadow_price as sp


class TestCh2(unittest.TestCase):
    def test_primal(self):
        x, z = sp.optimum()
        self.assertEqual((x, z), ((F(20), F(60)), 180))

    def test_finite_difference_matches_dual(self):
        self.assertEqual([sp.finite_difference(i) for i in range(3)], [1, 1, 0])

    def test_dual_certificate(self):
        feas, comp, dv, z = sp.check_dual_certificate((1, 1, 0))
        self.assertTrue(feas and comp)
        self.assertEqual(dv, z)

    def test_weak_duality_suboptimal_dual(self):
        feas, comp, dv, z = sp.check_dual_certificate((2, 0, 0))
        self.assertTrue(feas)
        self.assertGreater(dv, z)  # 쌍대 목적값은 주문제 최적값 이상

    def test_infeasible_dual(self):
        feas, _, _, _ = sp.check_dual_certificate((1, 0, 0))
        self.assertFalse(feas)

    def test_breakpoints_sewing(self):
        fine = sp.value_curve(range(70, 131))
        sl = sp.slopes(fine)
        changes = [(sl[i][0], sl[i - 1][2], sl[i][2]) for i in range(1, len(sl)) if sl[i][2] != sl[i - 1][2]]
        self.assertEqual(changes, [(80, 2, 1), (120, 1, 0)])

    def test_bump_beyond_range(self):
        self.assertEqual(sp.optimum(120)[1] - 180, 20)
        self.assertEqual(sp.optimum(130)[1] - 180, 20)


if __name__ == "__main__":
    unittest.main()
