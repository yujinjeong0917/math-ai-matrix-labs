import unittest

from request_path import MIB, PATHS, check, load_limits, step_table


class RequestPathTest(unittest.TestCase):
    def setUp(self):
        self.limits = load_limits()

    def test_limit_values_match_docs(self):
        self.assertEqual(self.limits["apigw_rest_integration_timeout_s"], 29)
        self.assertEqual(self.limits["apigw_http_integration_timeout_s"], 30)
        self.assertEqual(self.limits["lambda_sync_payload_mb"], 6)
        self.assertEqual(self.limits["sagemaker_invoke_body_bytes"], 6 * MIB)
        self.assertEqual(self.limits["sagemaker_invoke_timeout_s"], 60)

    def test_step_table_sums(self):
        rows = step_table(30000)
        self.assertEqual(sum(r["ms"] for r in rows), 30450)
        self.assertEqual(sum(r["ms"] for r in rows if r["scope"] == "inside"), 30250)

    def test_30s_model_fails_both_apis(self):
        for name in ("REST API + Lambda", "HTTP API + Lambda"):
            self.assertFalse(check(name, 30, 0.02, self.limits)["ok"], name)

    def test_max_model_seconds(self):
        self.assertAlmostEqual(check("REST API + Lambda", 1, 0.02, self.limits)["max_model_seconds"], 28.75)
        self.assertAlmostEqual(check("HTTP API + Lambda", 1, 0.02, self.limits)["max_model_seconds"], 29.75)

    def test_boundary(self):
        self.assertTrue(check("REST API + Lambda", 28.75, 0.02, self.limits)["ok"])
        self.assertFalse(check("REST API + Lambda", 28.8, 0.02, self.limits)["ok"])

    def test_payload_limit_is_lambda_6mb(self):
        r = check("HTTP API + Lambda", 20, 7, self.limits)
        self.assertFalse(r["ok"])
        self.assertEqual(r["size_key"], "lambda_sync_payload_mb")
        self.assertEqual(r["size_limit_bytes"], 6291456)
        self.assertTrue(check("HTTP API + Lambda", 20, 6, self.limits)["ok"])

    def test_sagemaker_path_uses_smaller_of_two_timeouts(self):
        r = check("REST API + Lambda + SageMaker 실시간", 20, 0.02, self.limits)
        self.assertEqual(r["tightest_time_key"], "apigw_rest_integration_timeout_s")

    def test_client_timeout(self):
        r = check("HTTP API + Lambda", 20, 0.02, self.limits, client_timeout_s=20)
        self.assertFalse(r["ok"])

    def test_paths_defined(self):
        self.assertEqual(len(PATHS), 3)


if __name__ == "__main__":
    unittest.main()
