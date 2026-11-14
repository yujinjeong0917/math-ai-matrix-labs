import unittest
from fractions import Fraction as F
import product_mix as p


class TestCh1(unittest.TestCase):
    def test_base(self):
        (x, v) = p.solve(p.BASE)
        self.assertEqual(x, (F(20), F(60)))
        self.assertEqual(v, 180)
        self.assertEqual(p.brute_force(p.BASE), ((20, 60), 180))

    def test_vertex_count(self):
        self.assertEqual(len(p.vertices(p.BASE)), 5)

    def test_hidden_minimum(self):
        cons = p.with_min_socks(70)
        x, v = p.solve(cons)
        self.assertEqual((x, v), ((F(10), F(70)), 170))
        self.assertEqual(p.brute_force(cons), ((10, 70), 170))
        # 숨은 조건을 빼고 푼 답은 이 조건을 어긴다
        self.assertLess(p.solve(p.BASE)[0][1], 70 + 0)  # 60 < 70

    def test_infeasible(self):
        cons = p.with_min_socks(90)
        self.assertIsNone(p.solve(cons))
        self.assertIsNone(p.brute_force(cons))


if __name__ == "__main__":
    unittest.main()
