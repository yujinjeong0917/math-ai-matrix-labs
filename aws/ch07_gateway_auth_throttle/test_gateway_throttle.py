import unittest

import gateway_throttle as g


class GatewayThrottleTest(unittest.TestCase):
    # 버킷 5개를 같은 순간(0ms)에 10건이 쓰면 5건만 통과
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

    # 오래 쉬어도 버킷은 burst(5개 = 5000밀리토큰)까지만 찬다
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

    # README 기대값: 손으로 따라간 결과(61번 시도, 마지막 성공 15.0초)와 일치
    def test_retry_without_jitter_matches_hand_calculation(self):
        r = g.retry_until_served(20, 2, 5, jitter=False)
        self.assertEqual(r, {"attempts": 61, "served": 20, "gave_up": 0, "last_ok_s": 15.0})

    # 시드가 같으면 결과가 같다. 숫자 69, 11.865는 파이썬 난수열에 의존하므로 파이썬 버전이 바뀌면 달라질 수 있음(확인 필요)
    def test_retry_with_jitter_is_deterministic(self):
        a = g.retry_until_served(20, 2, 5, jitter=True, seed=7)
        b = g.retry_until_served(20, 2, 5, jitter=True, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["served"], 20)
        self.assertEqual((a["attempts"], a["last_ok_s"]), (69, 11.865))

    # 시드를 바꿔도 시도 65~70번 안쪽(지터 없는 61번보다 많지만 마지막 성공은 더 빠를 수 있음)
    def test_jitter_seeds_1_to_8_range(self):
        rs = [g.retry_until_served(20, 2, 5, jitter=True, seed=s) for s in range(1, 9)]
        self.assertEqual((min(r["attempts"] for r in rs), max(r["attempts"] for r in rs)), (65, 70))
        self.assertEqual((min(r["last_ok_s"] for r in rs), max(r["last_ok_s"] for r in rs)), (11.125, 20.475))

    # 최대 시도 2번이면 일부는 포기하고, 성공 + 포기 = 20명
    def test_retries_give_up(self):
        r = g.retry_until_served(20, 2, 5, jitter=False, max_attempts=2)
        self.assertEqual(r["served"] + r["gave_up"], 20)
        self.assertGreater(r["gave_up"], 0)

    # 20000/86400 = 0.2315건/초, 피크 3배 0.6944, 모델 쪽 처리 한계 1.0은 그 1.44배
    def test_traffic_vs_limits(self):
        self.assertAlmostEqual(g.average_rps(20000), 0.2315, places=4)
        self.assertAlmostEqual(g.average_rps(20000) * 3, 0.6944, places=4)
        self.assertAlmostEqual(g.capacity_rps(30, 1, 30), 1.0)
        self.assertAlmostEqual(g.capacity_rps(30, 1, 30) / (g.average_rps(20000) * 3), 1.44, places=2)
        self.assertEqual(round(10000 / (g.average_rps(20000) * 3)), 14400)

    # REST 29초, HTTP 30초 "미만"이라 30초 모델은 둘 다 탈락, 28초(REST)와 29.5초(HTTP)는 통과. 11MB는 10MB 초과
    def test_sync_path_limits(self):
        self.assertTrue(g.sync_path_fits(30, "rest"))
        self.assertTrue(g.sync_path_fits(30, "http"))
        self.assertEqual(g.sync_path_fits(28, "rest"), [])
        self.assertEqual(g.sync_path_fits(29.5, "http"), [])
        self.assertTrue(g.sync_path_fits(5, "rest", payload_mb=11))


if __name__ == "__main__":
    unittest.main()
