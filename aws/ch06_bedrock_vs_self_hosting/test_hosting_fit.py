import json
import unittest

import hosting_fit as hf


def profile(**kw):
    """통과하는 기준 프로필(변환된 safetensors 모델, 미국 동부). 바꿀 항목만 kw로 덮어쓴다."""
    p = {"architecture": "Qwen2.5", "weights_format": "safetensors", "weights_gb": 15.2,
         "max_context_tokens": 32768, "region": "us-east-1", "needs_converse_api": False,
         "needs_batch_inference": False, "needs_cloudformation": False}
    p.update(kw)
    return p


class HostingFitTest(unittest.TestCase):
    # model_profile.json은 GGUF, 서울 리전, Qwen2.5의 Converse 요구라 막는 이유 3개(README)
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

    # 경계: 200GB는 탈락, 199.9GB는 통과. 문맥 131072는 탈락, 129000은 경고만(K의 정의 불명)
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

    # 시작 시 입력 + max_tokens, 끝난 뒤 입력 + 출력 x 소각률
    def test_token_quota_reserve_vs_final(self):
        self.assertEqual(hf.quota_reserve(800, 4096), 4896)
        self.assertEqual(hf.quota_reserve(800, 512), 1312)
        self.assertEqual(hf.quota_final(800, 300), 1100)
        self.assertEqual(hf.quota_final(1000, 100, 5), 1500)  # 문서의 예시 1,000 + 100 x 5

    # 100,000 // 4,896 = 20, 100,000 // 1,312 = 76 (쿼터는 가상)
    def test_concurrency_under_a_made_up_quota(self):
        self.assertEqual(hf.concurrent_by_tpm(100000, 800, 4096), 20)
        self.assertEqual(hf.concurrent_by_tpm(100000, 800, 512), 76)

    # 1시간 간격 20건이면 창 20개 x 4 CMU x 5분 = 400. 200초 안에 몰리면 1창
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
