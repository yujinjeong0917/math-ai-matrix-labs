import unittest
import routing as r


class RoutingTest(unittest.TestCase):
    def test_total_demand_and_min_vehicles(self):
        self.assertEqual(r.demand(r.STOPS), 25)
        self.assertEqual(r.solve(2)[0], None)  # 25박스 > 2대 x 10박스

    def test_exact_beats_heuristics(self):
        exact, routes = r.solve(3)
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
        self.assertEqual(r.one_big_route_cost()[0], 46.0)

    def test_time_windows(self):
        self.assertEqual(r.solve(3, r.WINDOWS_LOOSE)[0], 78.0)
        self.assertIsNone(r.solve(3, r.WINDOWS_TIGHT)[0])
        self.assertEqual(r.solve(4, r.WINDOWS_TIGHT)[0], 88.0)

    def test_slowdown(self):
        # 속도가 느려지면 거리는 같아도 시간창이 깨진다
        self.assertIsNone(r.solve(3, r.WINDOWS_LOOSE, 0.7)[0])
        self.assertAlmostEqual(r.solve(3, None, 0.7)[0], 76.0 / 0.7)

    def test_bigger_vehicle_and_search_size(self):
        self.assertEqual(r.solve(2, None, 1.0, 13)[0], 62.0)
        self.assertEqual(r.feasible_counts(), (85, 412))

    def test_deterministic(self):
        self.assertEqual(r.solve(3), r.solve(3))


if __name__ == "__main__":
    unittest.main()
