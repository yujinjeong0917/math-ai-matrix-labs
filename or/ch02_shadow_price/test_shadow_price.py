import unittest
from fractions import Fraction as F
import shadow_price as sp


class TestCh2(unittest.TestCase):
    def test_primal(self):
        x, z = sp.optimum()
        # 1장과 같은 기본 문제: 꼭짓점 (20, 60), 마진 180만원
        self.assertEqual((x, z), ((F(20), F(60)), 180))

    def test_finite_difference_matches_dual(self):
        # 한 단위 늘렸을 때 증가량 [1, 1, 0] = 그림자가격 y. 주문 한도는 최적해(x1=20)에서 남아 있어 0
        self.assertEqual([sp.finite_difference(i) for i in range(3)], [1, 1, 0])

    def test_dual_certificate(self):
        feas, comp, dv, z = sp.check_dual_certificate((1, 1, 0))
        self.assertTrue(feas and comp)
        self.assertEqual(dv, z)   # 강쌍대성: 최적이면 b.y = 180 = z

    def test_weak_duality_suboptimal_dual(self):
        feas, comp, dv, z = sp.check_dual_certificate((2, 0, 0))
        self.assertTrue(feas)
        self.assertGreater(dv, z)  # 쌍대 목적값은 주문제 최적값 이상

    def test_infeasible_dual(self):
        # (1,0,0): A^T y = (2, 1)이 c = (3, 2)에 못 미쳐 쌍대 가능성 위반
        feas, _, _, _ = sp.check_dual_certificate((1, 0, 0))
        self.assertFalse(feas)

    def test_breakpoints_sewing(self):
        # 재봉 시간 b1 = 70..130을 1시간 간격으로 훑어 기울기가 80에서 2->1, 120에서 1->0으로 바뀜을 확인(정수 간격이라 꺾임은 격자 점에서만 잡힌다)
        fine = sp.value_curve(range(70, 131))
        sl = sp.slopes(fine)
        changes = [(sl[i][0], sl[i - 1][2], sl[i][2]) for i in range(1, len(sl)) if sl[i][2] != sl[i - 1][2]]
        self.assertEqual(changes, [(80, 2, 1), (120, 1, 0)])

    def test_bump_beyond_range(self):
        # 기울기 1이 끝나는 b1=120에서 +20, 그 이상(130)에서도 +20: 그림자가격은 좁은 구간에서만 유효
        self.assertEqual(sp.optimum(120)[1] - 180, 20)
        self.assertEqual(sp.optimum(130)[1] - 180, 20)


if __name__ == "__main__":
    unittest.main()
