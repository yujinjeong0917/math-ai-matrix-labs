import json
import re
from pathlib import Path

import unittest

from hosting_filter import evaluate, fits_single_upload, gb_to_gib, load_candidates, peak_rpm, price_input_sheet

HERE = Path(__file__).parent


def profile(**updates):
    """기본 profile.json을 읽어 바꿀 항목만 덮어쓴 조건을 만든다."""
    base = json.loads((HERE / "profile.json").read_text(encoding="utf-8"))
    base.update(updates)
    return base


def status(result, cid):
    """후보 id의 최종 상태(BLOCKED 등)."""
    return result["candidates"][cid]["status"]


def check(result, cid, name):
    """후보 id의 검사 항목 하나(name)의 결과 사전."""
    return next(c for c in result["candidates"][cid]["checks"] if c["check"] == name)


class HostingFilterTest(unittest.TestCase):
    # 값이 있는 필드는 모두 출처 URL을 가져야 한다(출처 없는 숫자를 막는 장치)
    def test_every_known_value_has_a_source_url(self):
        for cid, cand in load_candidates().items():
            for name, field in cand["fields"].items():
                if field["value"] is not None:
                    assert field.get("source", "").startswith("https://"), (cid, name)


    # 가격을 코드와 데이터에 넣지 않는다는 원칙의 단순 점검: 통화 기호나 "숫자 + 원"이 없어야 한다(완전한 검사는 아님)
    def test_no_price_numbers_are_stored(self):
        text = (HERE / "candidates.json").read_text(encoding="utf-8")
        assert "$" not in text and "USD" not in text and not re.search(r"[0-9]\s*원", text)


    # README 기대 출력: Cloud Run CANDIDATE, Vercel BLOCKED(메모리 4GB < 8GB), Azure ML과 Cloudflare는 확인 못 한 항목이 남음
    def test_base_profile_splits_candidates_without_naming_a_winner(self):
        r = evaluate(profile())
        self.assertEqual(status(r, "cloud_run"), "CANDIDATE")
        self.assertEqual(status(r, "vercel"), "BLOCKED")
        self.assertEqual(status(r, "azure_ml"), "CANDIDATE_WITH_UNKNOWNS")
        self.assertEqual(status(r, "cloudflare_workers_ai"), "CANDIDATE_WITH_UNKNOWNS")
        assert "winner" not in r


    # README: GPU 기본 할당량 0, 메모리 문서의 32GiB와 80GiB 불일치가 주의 줄로 남는다
    def test_cloud_run_candidate_keeps_quota_and_memory_caveats(self):
        item = evaluate(profile())["candidates"]["cloud_run"]
        self.assertEqual(item["status"], "CANDIDATE")
        text = " ".join(item["caveats"])
        assert "할당량" in text and "32GiB" in text and "80GiB" in text


    # 무료만 쓰면 컨테이너 Space가 막히고, 문서로 무료 범위를 확인 못 한 Cloud Run은 UNKNOWN이라 후보 + 미확인으로 바뀐다
    def test_free_only_blocks_by_reason(self):
        r = evaluate(profile(free_only=True))
        self.assertEqual(check(r, "hf_spaces", "free_tier")["state"], "BLOCK")
        self.assertEqual(check(r, "vercel", "memory")["state"], "BLOCK")
        self.assertEqual(check(r, "cloud_run", "free_tier")["state"], "UNKNOWN")
        self.assertEqual(status(r, "cloud_run"), "CANDIDATE_WITH_UNKNOWNS")


    def test_every_candidate_keeps_manual_terms_check(self):
        for item in evaluate(profile())["candidates"].values():
            assert any("약관" in m for m in item["manual"])


    # 하루 요청 x 몰리는 비율(0.1)을 창 길이로 나눈 분당 값. 20건이면 2건이 한 창에 몰려 2/60, 2만 건이면 2000/60
    def test_peak_rpm_arithmetic(self):
        self.assertEqual(peak_rpm(profile(requests_per_day=20)), 20 * 0.1 / 60)
        self.assertEqual(peak_rpm(profile(requests_per_day=20000)), 2000 / 60)


    # 같은 후보도 부하에 따라 PASS(하한 이하), UNKNOWN(하한과 상한 사이), BLOCK(상한 초과)으로 갈린다
    def test_cloudflare_rate_limit_depends_on_load(self):
        low = evaluate(profile(requests_per_day=20))
        high = evaluate(profile(requests_per_day=20000))
        self.assertEqual(check(low, "cloudflare_workers_ai", "rate_limit")["state"], "PASS")
        self.assertEqual(check(high, "cloudflare_workers_ai", "rate_limit")["state"], "UNKNOWN")
        huge = evaluate(profile(requests_per_day=200000))
        self.assertEqual(check(huge, "cloudflare_workers_ai", "rate_limit")["state"], "BLOCK")


    # README: 허용 대기를 60초로 줄이면 시작 지연이 긴 후보가 BLOCKED(시작 + 모델 불러오기 > 60초)
    def test_short_wait_blocks_services_with_slow_start(self):
        r = evaluate(profile(acceptable_wait_seconds=60))
        self.assertEqual(check(r, "render", "startup")["state"], "BLOCK")
        self.assertEqual(check(r, "cloud_run", "startup")["state"], "BLOCK")
        self.assertEqual(status(r, "cloud_run"), "BLOCKED")


    # 600초 작업: 백그라운드 실행이 있는 후보(Netlify)는 비동기가 허용될 때만 통과, 일반 요청 제한만 있는 Azure ML은 BLOCK
    def test_long_job_uses_background_only_when_async_allowed(self):
        long_job = profile(seconds_per_request=600)
        r = evaluate(long_job)
        self.assertEqual(check(r, "netlify", "request_time")["state"], "PASS")
        self.assertEqual(check(r, "cloud_run", "request_time")["state"], "PASS")
        self.assertEqual(check(r, "azure_ml", "request_time")["state"], "BLOCK")
        r2 = evaluate(profile(seconds_per_request=600, async_allowed=False))
        self.assertEqual(check(r2, "netlify", "request_time")["state"], "BLOCK")


    # 위의 free_only 테스트와 비슷하지만 Render의 무료 범위가 PASS인지도 확인(항목이 겹침)
    def test_free_only_blocks_container_spaces(self):
        r = evaluate(profile(free_only=True))
        self.assertEqual(check(r, "hf_spaces", "free_tier")["state"], "BLOCK")
        self.assertEqual(check(r, "render", "free_tier")["state"], "PASS")
        self.assertEqual(check(r, "cloud_run", "free_tier")["state"], "UNKNOWN")


    # GPU가 필요 없으면 gpu 검사 항목 자체를 만들지 않는다
    def test_cpu_only_workload_does_not_need_gpu_checks(self):
        r = evaluate(profile(gpu_required=False))
        assert all(c["check"] != "gpu" for c in r["candidates"]["cloud_run"]["checks"])


    # README: 600건, 18,000초, 5시간, 720시간. 단가는 모두 None
    def test_price_sheet_has_quantities_but_no_prices(self):
        sheet = price_input_sheet(profile())
        q = sheet["quantities"]
        self.assertEqual(q["monthly_requests"], 600)
        self.assertEqual(q["active_seconds"], 18000)
        self.assertEqual(q["active_hours"], 5)
        self.assertEqual(q["always_on_hours"], 720)
        assert all(v["unit_price"] is None for v in sheet["per_candidate"].values())

    # 4.683GB는 4.361GiB로 5GiB 이내, 5.4GB는 5.03GiB라 한 번에 올릴 수 없다(경계 확인)
    def test_model_file_fits_single_r2_upload(self):
        self.assertAlmostEqual(gb_to_gib(4.683), 4.361, places=3)
        self.assertTrue(fits_single_upload(4.683))
        self.assertFalse(fits_single_upload(5.4))


if __name__ == "__main__":
    unittest.main()
