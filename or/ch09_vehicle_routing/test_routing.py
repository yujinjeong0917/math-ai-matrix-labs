import unittest
import routing as r


class RoutingTest(unittest.TestCase):
    def test_total_demand_and_min_vehicles(self):
        # 수요 합 25 > 2대 x 용량 10 = 20이므로 최대 2대로는 모든 점을 덮을 수 없다
        self.assertEqual(r.demand(r.STOPS), 25)
        self.assertEqual(r.solve(2)[0], None)  # 25박스 > 2대 x 10박스

    def test_exact_beats_heuristics(self):
        exact, routes = r.solve(3)
        # 전수 탐색 76 < 단순 순서 78 < 가까운 곳 먼저 82. 거리는 정수 좌표의 맨해튼 합이라 float 비교도 정확하다
        self.assertEqual(exact, 76.0)
        self.assertEqual(r.sweep_routes()[0], 78.0)
        self.assertEqual(r.nearest_neighbor()[0], 82.0)
        self.assertLess(exact, r.sweep_routes()[0])
        self.assertLess(exact, r.nearest_neighbor()[0])
        # 모든 정지점을 정확히 한 번 방문, 용량 준수
        visited = sorted(s for rt in routes for s in rt)
        self.assertEqual(visited, sorted(r.STOPS))
        for rt in routes:
            self.assertLessEqual(r.demand(rt), r.CAPACITY)

    def test_lower_bound_single_route(self):
        # 용량을 무시하고 한 대가 도는 최단 경로: 여러 대 해(76)의 하한 비교용
        self.assertEqual(r.one_big_route_cost()[0], 46.0)

    def test_time_windows(self):
        # 시간창이 하나 걸리면 76 -> 78, 빡빡하면 3대로는 불가능하고 4대로 88(이동거리가 늘어난다)
        self.assertEqual(r.solve(3, r.WINDOWS_LOOSE)[0], 78.0)
        self.assertIsNone(r.solve(3, r.WINDOWS_TIGHT)[0])
        self.assertEqual(r.solve(4, r.WINDOWS_TIGHT)[0], 88.0)

    def test_slowdown(self):
        # 속도가 느려지면 거리는 같아도 시간창이 깨진다
        self.assertIsNone(r.solve(3, r.WINDOWS_LOOSE, 0.7)[0])
        # 시간창이 없으면 거리 76이 0.7배 속도에서 분 단위로 76/0.7. 부동소수점이라 AlmostEqual
        self.assertAlmostEqual(r.solve(3, None, 0.7)[0], 76.0 / 0.7)

    def test_bigger_vehicle_and_search_size(self):
        self.assertEqual(r.solve(2, None, 1.0, 13)[0], 62.0)
        # 용량 10 이하 묶음 85개, 그 안의 방문 순서 총합(k!의 합) 412: 전수 탐색이 실제로 훑는 규모
        self.assertEqual(r.feasible_counts(), (85, 412))

    def test_deterministic(self):
        self.assertEqual(r.solve(3), r.solve(3))


if __name__ == "__main__":
    unittest.main()
