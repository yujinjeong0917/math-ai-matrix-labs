import unittest

import gateway_throttle as g


class GatewayThrottleTest(unittest.TestCase):
    def test_burst_then_429(self):
        b = g.TokenBucket(2, 5)
        codes = g.simulate([0] * 10, [b])
        self.assertEqual(codes.count(200), 5)
        self.assertEqual(codes.count(429), 5)

    def test_refill_rate(self):
        b = g.TokenBucket(2, 5)
        g.simulate([0] * 5, [b])
        self.assertFalse(b.allow(0))
        self.assertFalse(b.allow(499))
        self.assertTrue(b.allow(500))      # 초당 2개이면 0.5초에 1개
        self.assertFalse(b.allow(500))
        self.assertTrue(b.allow(1000))

    def test_bucket_never_exceeds_burst(self):
        b = g.TokenBucket(2, 5)
        self.assertTrue(b.has_token(60000))
        self.assertEqual(b.tokens, 5000)

    def test_tightest_layer_decides(self):
        client = g.TokenBucket(1, 2)       # 사용량 계획의 클라이언트 제한
        account = g.TokenBucket(100, 100)  # 계정 제한은 넉넉
        codes = g.simulate([0] * 5, [client, account])
        self.assertEqual(codes.count(200), 2)
        self.assertEqual(account.tokens, 98000)  # 거절된 요청은 계정 토큰을 쓰지 않음

    def test_daily_quota(self):
        q = g.DailyQuota(3)
        self.assertEqual([q.allow() for _ in range(5)], [200, 200, 200, 429, 429])

    def test_retry_without_jitter_matches_hand_calculation(self):
        r = g.retry_until_served(20, 2, 5, jitter=False)
        self.assertEqual(r, {"attempts": 61, "served": 20, "gave_up": 0, "last_ok_s": 15.0})

    def test_retry_with_jitter_is_deterministic(self):
        a = g.retry_until_served(20, 2, 5, jitter=True, seed=7)
        b = g.retry_until_served(20, 2, 5, jitter=True, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["served"], 20)
        self.assertEqual((a["attempts"], a["last_ok_s"]), (69, 11.865))

    def test_jitter_seeds_1_to_8_range(self):
        rs = [g.retry_until_served(20, 2, 5, jitter=True, seed=s) for s in range(1, 9)]
        self.assertEqual((min(r["attempts"] for r in rs), max(r["attempts"] for r in rs)), (65, 70))
        self.assertEqual((min(r["last_ok_s"] for r in rs), max(r["last_ok_s"] for r in rs)), (11.125, 20.475))

    def test_retries_give_up(self):
        r = g.retry_until_served(20, 2, 5, jitter=False, max_attempts=2)
        self.assertEqual(r["served"] + r["gave_up"], 20)
        self.assertGreater(r["gave_up"], 0)

    def test_traffic_vs_limits(self):
        self.assertAlmostEqual(g.average_rps(20000), 0.2315, places=4)
        self.assertAlmostEqual(g.average_rps(20000) * 3, 0.6944, places=4)
        self.assertAlmostEqual(g.capacity_rps(30, 1, 30), 1.0)
        self.assertAlmostEqual(g.capacity_rps(30, 1, 30) / (g.average_rps(20000) * 3), 1.44, places=2)
        self.assertEqual(round(10000 / (g.average_rps(20000) * 3)), 14400)

    def test_sync_path_limits(self):
        self.assertTrue(g.sync_path_fits(30, "rest"))
        self.assertTrue(g.sync_path_fits(30, "http"))
        self.assertEqual(g.sync_path_fits(28, "rest"), [])
        self.assertEqual(g.sync_path_fits(29.5, "http"), [])
        self.assertTrue(g.sync_path_fits(5, "rest", payload_mb=11))


if __name__ == "__main__":
    unittest.main()
