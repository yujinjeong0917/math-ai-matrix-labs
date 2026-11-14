import json
import unittest
from pathlib import Path

import cost_inputs as c

PROFILE = json.loads(Path(__file__).with_name("cost_profile.json").read_text(encoding="utf-8"))


class CostInputsTest(unittest.TestCase):
    def test_workers_needed_small_and_scaled(self):
        self.assertEqual(c.workers_needed(20, 30, 0.5), 1)
        self.assertEqual(c.workers_needed(20000, 30, 0.5), 14)

    def test_idle_share_small(self):
        r = c.idle_share(20, 30, 1, 30)
        self.assertAlmostEqual(r["busy_hours"], 5)
        self.assertAlmostEqual(r["always_on_hours"], 720)
        self.assertAlmostEqual(r["idle_share"], 1 - 5 / 720)

    def test_idle_share_scaled(self):
        r = c.idle_share(20000, 30, 14, 30)
        self.assertAlmostEqual(r["busy_hours"], 5000)
        self.assertAlmostEqual(r["always_on_hours"], 10080)
        self.assertAlmostEqual(r["idle_share"], 5080 / 10080)

    def test_log_volume_and_retention(self):
        q = c.quantities(PROFILE, "scaled")
        self.assertAlmostEqual(q["prompt_log_gb_per_day"], 0.4)
        self.assertAlmostEqual(q["metadata_log_gb_per_day"], 0.01)
        self.assertAlmostEqual(q["log_stored_gb"], 12.0)
        self.assertAlmostEqual(q["log_stored_gb_no_expiry"], 144.0)
        self.assertAlmostEqual(q["log_stored_gb_no_expiry"] / q["log_stored_gb"], 12.0)

    def test_nat_and_storage_versions(self):
        q = c.quantities(PROFILE, "scaled")
        self.assertAlmostEqual(q["nat_gb"], 280.98, places=2)
        self.assertAlmostEqual(q["model_storage_gb"], 46.83, places=2)
        self.assertAlmostEqual(q["model_storage_gb_lifecycle"], 14.049, places=3)
        self.assertAlmostEqual(q["image_storage_gb"], 60)
        self.assertAlmostEqual(q["image_storage_gb_lifecycle"], 18)

    def test_data_out(self):
        self.assertAlmostEqual(c.quantities(PROFILE, "scaled")["data_out_gb"], 1.8)
        self.assertAlmostEqual(c.quantities(PROFILE, "small")["data_out_gb"], 0.0018)

    def test_missing_prices_are_hold(self):
        q = c.quantities(PROFILE, "small")
        lines = c.cost_lines(q, PROFILE["unit_prices"])
        self.assertTrue(all(v == "HOLD" for v in lines.values()))

    def test_cost_line_is_quantity_times_price(self):
        q = c.quantities(PROFILE, "scaled")
        prices = dict(PROFILE["unit_prices"], nat_gb=2.0)  # 2.0은 계산 확인용 가짜 숫자
        lines = c.cost_lines(q, prices)
        self.assertAlmostEqual(lines["nat_gb"], 2.0 * q["nat_gb"])
        self.assertEqual(lines["compute_hours"], "HOLD")

    def test_profile_has_no_real_prices(self):
        self.assertTrue(all(v is None for v in PROFILE["unit_prices"].values()))

    def test_invalid_utilization(self):
        with self.assertRaises(ValueError):
            c.workers_needed(20, 30, 0)


if __name__ == "__main__":
    unittest.main()
