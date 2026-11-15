import unittest

import endpoint_modes as em


class EndpointModesTest(unittest.TestCase):
    # 20000/86400 x 30 = 6.9444. 부동소수점이라 근사 비교. 하루 20건 x 30초 = 600초
    def test_littles_law_for_20k_per_day(self):
        self.assertAlmostEqual(em.average_concurrency(20000, 30), 6.9444, places=3)
        self.assertEqual(em.average_concurrency(20, 30) * 86400, 600)  # 하루 600초

    # 7대(평균 6.94 올림), /0.7 -> 9.92 -> 10, 피크 3배 -> 29.76 -> 30, 인스턴스당 4건 -> 7.44 -> 8
    def test_instances_for_peak(self):
        self.assertEqual(em.instances_needed(20000, 30), 7)
        self.assertEqual(em.instances_needed(20000, 30, 1, 0.7), 10)
        self.assertEqual(em.instances_needed(20000, 30, 3, 0.7), 30)
        self.assertEqual(em.instances_needed(20000, 30, 3, 0.7, 4), 8)
        self.assertEqual(em.monthly_instance_hours(1), 720)
        self.assertEqual(em.monthly_instance_hours(30), 21600)

    # 경계: 실시간 60초·6MB는 "미만"이라 75초와 정확히 6MB가 탈락, 비동기 1,100MB는 1,024MB 초과
    def test_limits_follow_official_matrix(self):
        self.assertEqual(em.check_limits("realtime", 0.02, 45, True), [])
        self.assertTrue(em.check_limits("realtime", 0.02, 75, True))      # 60초 미만이어야 함
        self.assertTrue(em.check_limits("realtime", 6, 30, True))         # 6MB 미만이어야 함
        self.assertEqual(em.check_limits("async", 500, 1800, True), [])
        self.assertTrue(em.check_limits("async", 1100, 30, True))
        self.assertIn("GPU 미지원", em.check_limits("serverless", 0.02, 30, True))
        self.assertEqual(em.check_limits("batch", 100, 90000, True), [])

    # 20건 동시 도착: 90 + 30 x 순번. 4대면 5번째 차례까지라 20번째가 90 + 5 x 30 = 240
    def test_async_burst_of_20(self):
        one = em.async_burst_completions(20, 30, 90, 1)
        self.assertEqual([one[0], one[9], one[19]], [120, 390, 690])  # 4장 표와 같은 값
        four = em.async_burst_completions(20, 30, 90, 4)
        self.assertEqual([four[0], four[9], four[19]], [120, 180, 240])

    # 20/5 = 4대, 100/5 = 20이지만 최대 5대로 제한, 백로그 0이면 0대
    def test_backlog_target_tracking_sketch(self):
        self.assertEqual(em.desired_instances(20, 5, 5), 4)
        self.assertEqual(em.desired_instances(100, 5, 5), 5)
        self.assertEqual(em.desired_instances(0, 5, 5), 0)

    # 따로 오면 20 x 420 = 8,400초, 몰리면 90 + 600 + 300 = 990초
    def test_active_seconds(self):
        self.assertEqual(em.async_active_seconds_isolated(20, 30, 90, 300), 8400)
        self.assertAlmostEqual(8400 / 3600, 2.3333, places=3)
        self.assertEqual(em.async_active_seconds_burst(20, 30, 90, 300), 990)
        self.assertAlmostEqual(8400 * 30 / 3600, 70.0)       # 월 가동 시간(따로 올 때)
        self.assertAlmostEqual(990 * 30 / 3600, 8.25)         # 월 가동 시간(하루 한 번 몰릴 때)

    # 20000 x 30 = 600,000초 = 166.67시간(README). 30개 파일 30대면 ceil(20000/30) = 667차례
    def test_batch_one_file_uses_one_instance(self):
        span, workers = em.batch_makespan_seconds(20000, 30, 1, 4)
        self.assertEqual((span, workers), (600000, 1))
        span, workers = em.batch_makespan_seconds(20000, 30, 4, 4)
        self.assertEqual((span, workers), (150000, 4))
        span, workers = em.batch_makespan_seconds(20000, 30, 30, 30)
        self.assertEqual(span, 667 * 30)

    # 4 x 25 = 100MB는 허용, 5 x 25 = 125MB와 건당 101MB는 탈락
    def test_batch_payload_product(self):
        self.assertTrue(em.batch_config_ok(4, 25))
        self.assertFalse(em.batch_config_ok(5, 25))
        self.assertFalse(em.batch_config_ok(1, 101))

    # README 기대 출력과 같은 값. 실행 위치의 endpoint_profile.json을 상대 경로로 읽는다(이 폴더에서 실행)
    def test_evaluate_profile(self):
        import json
        with open("endpoint_profile.json", encoding="utf-8") as f:
            r = em.evaluate(json.load(f))
        self.assertEqual(r["instances_for_peak"], 30)
        self.assertEqual(r["always_on_instance_hours_month"], 21600)
        self.assertIn("GPU 미지원", r["limit_failures"]["serverless"])
        self.assertEqual(r["limit_failures"]["realtime"], [])
        self.assertEqual(r["batch_workers"], 1)


if __name__ == "__main__":
    unittest.main()
