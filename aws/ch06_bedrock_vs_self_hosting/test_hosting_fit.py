import json
import unittest

import hosting_fit as hf


def profile(**kw):
    p = {"architecture": "Qwen2.5", "weights_format": "safetensors", "weights_gb": 15.2,
         "max_context_tokens": 32768, "region": "us-east-1", "needs_converse_api": False,
         "needs_batch_inference": False, "needs_cloudformation": False}
    p.update(kw)
    return p


class HostingFitTest(unittest.TestCase):
    def test_chapter_profile_is_blocked_three_ways(self):
        with open("model_profile.json", encoding="utf-8") as f:
            r = hf.run(json.load(f))
        self.assertFalse(r["cmi_possible"])
        self.assertEqual(len(r["blockers"]), 3)  # gguf, 서울 리전, Converse
        self.assertTrue(any("gguf" in b for b in r["blockers"]))
        self.assertTrue(any("ap-northeast-2" in b for b in r["blockers"]))
        self.assertTrue(any("Converse" in b for b in r["blockers"]))

    def test_converted_model_in_supported_region_passes(self):
        blockers, notes = hf.check_cmi(profile())
        self.assertEqual(blockers, [])
        self.assertTrue(notes)

    def test_limits(self):
        self.assertTrue(hf.check_cmi(profile(weights_gb=200))[0])
        self.assertEqual(hf.check_cmi(profile(weights_gb=199.9))[0], [])
        self.assertTrue(hf.check_cmi(profile(max_context_tokens=131072))[0])
        self.assertIn("확인 필요", hf.check_cmi(profile(max_context_tokens=129000))[1][0])
        self.assertTrue(hf.check_cmi(profile(architecture="Gemma"))[0])
        self.assertTrue(hf.check_cmi(profile(needs_batch_inference=True))[0])
        self.assertEqual(hf.check_cmi(profile(architecture="Llama", needs_converse_api=True))[0], [])
        self.assertEqual(hf.check_cmi(profile(architecture="Qwen2", needs_converse_api=True))[0], [])
        self.assertTrue(hf.check_cmi(profile(architecture="Qwen2.5", needs_converse_api=True))[0])

    def test_token_quota_reserve_vs_final(self):
        self.assertEqual(hf.quota_reserve(800, 4096), 4896)
        self.assertEqual(hf.quota_reserve(800, 512), 1312)
        self.assertEqual(hf.quota_final(800, 300), 1100)
        self.assertEqual(hf.quota_final(1000, 100, 5), 1500)  # 문서의 예시 1,000 + 100 x 5

    def test_concurrency_under_a_made_up_quota(self):
        self.assertEqual(hf.concurrent_by_tpm(100000, 800, 4096), 20)
        self.assertEqual(hf.concurrent_by_tpm(100000, 800, 512), 76)

    def test_billing_windows_lower_bound(self):
        self.assertEqual(hf.billing_windows([]), 0)
        self.assertEqual(hf.billing_windows([0, 10, 299]), 1)
        self.assertEqual(hf.billing_windows([0, 300]), 2)
        spread = [i * 3600 for i in range(20)]  # 20건이 1시간 간격
        self.assertEqual(hf.billing_windows(spread), 20)
        self.assertEqual(hf.cmu_minutes(1, 4, 20), 400)
        self.assertEqual(hf.cmu_minutes(2, 4, 1), 40)
        self.assertEqual(hf.cmu_minutes(1, 4, 1), 20)
        self.assertEqual(hf.billing_windows([i * 10 for i in range(20)]), 1)  # 20건이 200초 안에 몰림


if __name__ == "__main__":
    unittest.main()
