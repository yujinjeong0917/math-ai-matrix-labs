import json
import unittest
from pathlib import Path

from policy_scan import allowed, arn_match, risk_score, scan

P = json.loads((Path(__file__).parent / "policies.json").read_text(encoding="utf-8"))


class ScanTest(unittest.TestCase):
    def test_too_broad(self):
        f = scan(P["too_broad"])
        self.assertEqual(sorted(x["level"] for x in f), ["HIGH", "HIGH", "REVIEW"])
        self.assertEqual(risk_score(f), 7)

    def test_half_broad(self):
        f = scan(P["half_broad"])
        self.assertEqual(sum(x["level"] == "HIGH" for x in f), 1)
        self.assertEqual(sum(x["level"] == "REVIEW" for x in f), 5)
        self.assertEqual(risk_score(f), 8)

    def test_scoped_is_clean(self):
        self.assertEqual(scan(P["scoped"]), [])
        self.assertEqual(risk_score([]), 0)

    def test_deny_statements_are_not_flagged(self):
        pol = {"Statement": [{"Effect": "Deny", "Action": "*", "Resource": "*"}]}
        self.assertEqual(scan(pol), [])


class EvalTest(unittest.TestCase):
    def test_default_deny_and_explicit_deny(self):
        s = P["scoped"]
        self.assertEqual(allowed(s, "s3:GetObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf"), "ALLOW")
        self.assertEqual(allowed(s, "s3:GetObject", "arn:aws:s3:::example-model-bucket/other/secret.txt"), "DENY(기본)")
        self.assertEqual(allowed(s, "s3:DeleteObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf"), "DENY(명시)")

    def test_explicit_deny_beats_allow(self):
        pol = {"Statement": [
            {"Effect": "Allow", "Action": "s3:*", "Resource": "*"},
            {"Effect": "Deny", "Action": "s3:DeleteObject", "Resource": "*"}]}
        self.assertEqual(allowed(pol, "s3:DeleteObject", "arn:aws:s3:::b/k"), "DENY(명시)")
        self.assertEqual(allowed(pol, "s3:GetObject", "arn:aws:s3:::b/k"), "ALLOW")

    def test_action_is_case_insensitive(self):
        pol = {"Statement": [{"Effect": "Allow", "Action": "IAM:listaccesskeys", "Resource": "*"}]}
        self.assertEqual(allowed(pol, "iam:ListAccessKeys", "arn:aws:iam::1:user/x"), "ALLOW")

    def test_doc_wildcard_example(self):
        pat = "arn:aws:s3:::amzn-s3-demo-bucket/*/test/*"
        b = "arn:aws:s3:::amzn-s3-demo-bucket/"
        for ok in ["1/test/object.jpg", "1/2/test/object.jpg", "1/2/test/3/object.jpg", "1/2/3/test/4/object.jpg",
                   "1///test///object.jpg", "1/test/.jpg", "/test/object.jpg", "1/test/"]:
            self.assertTrue(arn_match(pat, b + ok), ok)
        for no in ["1-test/object.jpg", "test/object.jpg", "1/2/test.jpg"]:
            self.assertFalse(arn_match(pat, b + no), no)


if __name__ == "__main__":
    unittest.main()
