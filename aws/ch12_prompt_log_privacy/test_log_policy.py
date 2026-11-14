import json
import unittest
from pathlib import Path

import log_policy as lp

CFG = json.loads(Path(__file__).with_name("log_config.json").read_text(encoding="utf-8"))


class LogPolicyTest(unittest.TestCase):
    def test_masks_four_kinds(self):
        text = "010-1234-5678 test@example.com 1234-5678-9012-3456 900101-1234567"
        masked, counts = lp.mask_text(text)
        self.assertEqual(masked, "[전화] [이메일] [카드번호형] [주민번호형]")
        self.assertEqual(counts, {"주민번호형": 1, "카드번호형": 1, "전화": 1, "이메일": 1})

    def test_free_text_name_is_not_caught(self):
        masked, counts = lp.mask_text("홍길동 고객님 서울시 강남구 어딘가")
        self.assertEqual(masked, "홍길동 고객님 서울시 강남구 어딘가")
        self.assertEqual(counts, {})

    def test_allowed_retention_list(self):
        self.assertIn(7, lp.ALLOWED_RETENTION_DAYS)
        self.assertIn(30, lp.ALLOWED_RETENTION_DAYS)
        self.assertNotIn(10, lp.ALLOWED_RETENTION_DAYS)
        self.assertNotIn(45, lp.ALLOWED_RETENTION_DAYS)
        self.assertEqual(len(lp.ALLOWED_RETENTION_DAYS), 22)

    def test_worst_case_age(self):
        self.assertEqual(lp.worst_case_age_days(7), 10)

    def test_stored_gb(self):
        self.assertAlmostEqual(lp.stored_gb(20000, 20000, 7), 2.8)
        self.assertAlmostEqual(lp.stored_gb(20000, 20000, 90), 36.0)
        self.assertAlmostEqual(lp.stored_gb(20000, 20000, 7, 10), 0.28)

    def test_unmasked_backlog(self):
        self.assertEqual(lp.unmasked_events(20000, 3), 60000)

    def test_config_findings(self):
        f = lp.check_log_groups(CFG)
        self.assertEqual(len(f), 2)
        self.assertTrue(any("/app/answers/debug" in x and "만료 없음" in x for x in f))
        self.assertFalse(any("/app/answers/raw-prompts" in x for x in f))
        self.assertFalse(any("/app/answers/metadata" in x for x in f))

    def test_bad_retention_value(self):
        cfg = dict(CFG, log_groups=[{"name": "g", "retention_days": 10, "contains_raw_prompt": False}])
        self.assertTrue(any("허용된 보관 기간이 아니에요" in x for x in lp.check_log_groups(cfg)))

    def test_account_policy_overrides_group(self):
        cfg = dict(CFG, account_policy_days=90)
        f = lp.check_log_groups(cfg)
        self.assertTrue(any("raw-prompts" in x and "덮어써요" in x for x in f))
        self.assertTrue(any("raw-prompts" in x and "90일" in x and "허용 7일" in x for x in f))

    def test_account_policy_minimum(self):
        cfg = dict(CFG, account_policy_days=14)
        self.assertTrue(any("최소 30일" in x for x in lp.check_log_groups(cfg)))


if __name__ == "__main__":
    unittest.main()
