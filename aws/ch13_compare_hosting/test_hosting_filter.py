import json
import re
from pathlib import Path

import unittest

from hosting_filter import evaluate, fits_single_upload, gb_to_gib, load_candidates, peak_rpm, price_input_sheet

HERE = Path(__file__).parent


def profile(**updates):
    base = json.loads((HERE / "profile.json").read_text(encoding="utf-8"))
    base.update(updates)
    return base


def status(result, cid):
    return result["candidates"][cid]["status"]


def check(result, cid, name):
    return next(c for c in result["candidates"][cid]["checks"] if c["check"] == name)


class HostingFilterTest(unittest.TestCase):
    def test_every_known_value_has_a_source_url(self):
        for cid, cand in load_candidates().items():
            for name, field in cand["fields"].items():
                if field["value"] is not None:
                    assert field.get("source", "").startswith("https://"), (cid, name)


    def test_no_price_numbers_are_stored(self):
        text = (HERE / "candidates.json").read_text(encoding="utf-8")
        assert "$" not in text and "USD" not in text and not re.search(r"[0-9]\s*원", text)


    def test_base_profile_splits_candidates_without_naming_a_winner(self):
        r = evaluate(profile())
        self.assertEqual(status(r, "cloud_run"), "CANDIDATE")
        self.assertEqual(status(r, "vercel"), "BLOCKED")
        self.assertEqual(status(r, "azure_ml"), "CANDIDATE_WITH_UNKNOWNS")
        self.assertEqual(status(r, "cloudflare_workers_ai"), "CANDIDATE_WITH_UNKNOWNS")
        assert "winner" not in r


    def test_cloud_run_candidate_keeps_quota_and_memory_caveats(self):
        item = evaluate(profile())["candidates"]["cloud_run"]
        self.assertEqual(item["status"], "CANDIDATE")
        text = " ".join(item["caveats"])
        assert "할당량" in text and "32GiB" in text and "80GiB" in text


    def test_free_only_blocks_by_reason(self):
        r = evaluate(profile(free_only=True))
        self.assertEqual(check(r, "hf_spaces", "free_tier")["state"], "BLOCK")
        self.assertEqual(check(r, "vercel", "memory")["state"], "BLOCK")
        self.assertEqual(check(r, "cloud_run", "free_tier")["state"], "UNKNOWN")
        self.assertEqual(status(r, "cloud_run"), "CANDIDATE_WITH_UNKNOWNS")


    def test_every_candidate_keeps_manual_terms_check(self):
        for item in evaluate(profile())["candidates"].values():
            assert any("약관" in m for m in item["manual"])


    def test_peak_rpm_arithmetic(self):
        self.assertEqual(peak_rpm(profile(requests_per_day=20)), 20 * 0.1 / 60)
        self.assertEqual(peak_rpm(profile(requests_per_day=20000)), 2000 / 60)


    def test_cloudflare_rate_limit_depends_on_load(self):
        low = evaluate(profile(requests_per_day=20))
        high = evaluate(profile(requests_per_day=20000))
        self.assertEqual(check(low, "cloudflare_workers_ai", "rate_limit")["state"], "PASS")
        self.assertEqual(check(high, "cloudflare_workers_ai", "rate_limit")["state"], "UNKNOWN")
        huge = evaluate(profile(requests_per_day=200000))
        self.assertEqual(check(huge, "cloudflare_workers_ai", "rate_limit")["state"], "BLOCK")


    def test_short_wait_blocks_services_with_slow_start(self):
        r = evaluate(profile(acceptable_wait_seconds=60))
        self.assertEqual(check(r, "render", "startup")["state"], "BLOCK")
        self.assertEqual(check(r, "cloud_run", "startup")["state"], "BLOCK")
        self.assertEqual(status(r, "cloud_run"), "BLOCKED")


    def test_long_job_uses_background_only_when_async_allowed(self):
        long_job = profile(seconds_per_request=600)
        r = evaluate(long_job)
        self.assertEqual(check(r, "netlify", "request_time")["state"], "PASS")
        self.assertEqual(check(r, "cloud_run", "request_time")["state"], "PASS")
        self.assertEqual(check(r, "azure_ml", "request_time")["state"], "BLOCK")
        r2 = evaluate(profile(seconds_per_request=600, async_allowed=False))
        self.assertEqual(check(r2, "netlify", "request_time")["state"], "BLOCK")


    def test_free_only_blocks_container_spaces(self):
        r = evaluate(profile(free_only=True))
        self.assertEqual(check(r, "hf_spaces", "free_tier")["state"], "BLOCK")
        self.assertEqual(check(r, "render", "free_tier")["state"], "PASS")
        self.assertEqual(check(r, "cloud_run", "free_tier")["state"], "UNKNOWN")


    def test_cpu_only_workload_does_not_need_gpu_checks(self):
        r = evaluate(profile(gpu_required=False))
        assert all(c["check"] != "gpu" for c in r["candidates"]["cloud_run"]["checks"])


    def test_price_sheet_has_quantities_but_no_prices(self):
        sheet = price_input_sheet(profile())
        q = sheet["quantities"]
        self.assertEqual(q["monthly_requests"], 600)
        self.assertEqual(q["active_seconds"], 18000)
        self.assertEqual(q["active_hours"], 5)
        self.assertEqual(q["always_on_hours"], 720)
        assert all(v["unit_price"] is None for v in sheet["per_candidate"].values())

    def test_model_file_fits_single_r2_upload(self):
        self.assertAlmostEqual(gb_to_gib(4.683), 4.361, places=3)
        self.assertTrue(fits_single_upload(4.683))
        self.assertFalse(fits_single_upload(5.4))


if __name__ == "__main__":
    unittest.main()
