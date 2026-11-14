import unittest
import local_global as lg


class TestCh4(unittest.TestCase):
    def test_two_local_maxima(self):
        _, local = lg.grid_scan()
        self.assertEqual(len(local), 2)
        self.assertAlmostEqual(local[0][0], 41.58, places=1)
        self.assertAlmostEqual(local[1][0], 86.16, places=1)

    def test_starts_reach_different_optima(self):
        a = lg.ascend(25)
        b = lg.ascend(80)
        self.assertAlmostEqual(a[0], 41.58, places=1)
        self.assertAlmostEqual(b[0], 86.16, places=1)
        self.assertLess(a[1], b[1])
        self.assertAlmostEqual(a[1], 1707.22, places=1)
        self.assertAlmostEqual(b[1], 1805.19, places=1)

    def test_both_stops_are_stationary_maxima(self):
        for s in (25, 80):
            p, _, _ = lg.ascend(s)
            self.assertLess(abs(lg.grad(p)), 1e-3)
            self.assertLess(lg.curvature(p), 0)

    def test_multistart_matches_grid_max(self):
        pts, _ = lg.grid_scan()
        gbest = max(pts, key=lambda t: t[1])
        _, best = lg.multi_start([25, 40, 60, 80, 110])
        self.assertAlmostEqual(best[1], gbest[1], places=1)

    def test_single_start_can_miss(self):
        pts, _ = lg.grid_scan()
        gbest = max(v for _, v in pts)
        self.assertLess(lg.ascend(40)[1], gbest - 50)

    def test_deterministic(self):
        self.assertEqual(lg.ascend(60), lg.ascend(60))


if __name__ == "__main__":
    unittest.main()
