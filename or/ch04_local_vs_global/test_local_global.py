import unittest
import local_global as lg


class TestCh4(unittest.TestCase):
    def test_two_local_maxima(self):
        # 0.01 간격 격자에서 국소 최대가 정확히 둘: 약 41.58과 86.16(README 기대 출력). places=1은 격자 간격 0.01보다 넉넉한 허용 오차
        _, local = lg.grid_scan()
        self.assertEqual(len(local), 2)
        self.assertAlmostEqual(local[0][0], 41.58, places=1)
        self.assertAlmostEqual(local[1][0], 86.16, places=1)

    def test_starts_reach_different_optima(self):
        # 시작 25는 왼쪽 봉우리(이익 1707.22), 시작 80은 오른쪽 봉우리(1805.19)에 멈춘다: 시작점에 따라 답이 다르다
        a = lg.ascend(25)
        b = lg.ascend(80)
        self.assertAlmostEqual(a[0], 41.58, places=1)
        self.assertAlmostEqual(b[0], 86.16, places=1)
        self.assertLess(a[1], b[1])
        self.assertAlmostEqual(a[1], 1707.22, places=1)
        self.assertAlmostEqual(b[1], 1805.19, places=1)

    def test_both_stops_are_stationary_maxima(self):
        # 멈춘 점이 기울기 약 0이면서 곡률 < 0(봉우리)임을 확인; 골짜기에서 멈춘 것이 아님
        for s in (25, 80):
            p, _, _ = lg.ascend(s)
            self.assertLess(abs(lg.grad(p)), 1e-3)
            self.assertLess(lg.curvature(p), 0)

    def test_multistart_matches_grid_max(self):
        # 다중 시작의 최선이 격자 전역 최대와 일치(둘 다 약 1805.19)
        pts, _ = lg.grid_scan()
        gbest = max(pts, key=lambda t: t[1])
        _, best = lg.multi_start([25, 40, 60, 80, 110])
        self.assertAlmostEqual(best[1], gbest[1], places=1)

    def test_single_start_can_miss(self):
        pts, _ = lg.grid_scan()
        gbest = max(v for _, v in pts)
        # 시작 40은 왼쪽 봉우리(1707.22)에 갇혀 전역 최대 1805.19보다 약 98 낮다. 여유 50은 허용 마진
        self.assertLess(lg.ascend(40)[1], gbest - 50)

    def test_deterministic(self):
        # 난수 없이 결정적이라 같은 시작점은 정확히 같은 결과(시드 고정이 필요 없다)
        self.assertEqual(lg.ascend(60), lg.ascend(60))


if __name__ == "__main__":
    unittest.main()
