import json
import unittest
from pathlib import Path

from policy_scan import allowed, arn_match, risk_score, scan

P = json.loads((Path(__file__).parent / "policies.json").read_text(encoding="utf-8"))


class ScanTest(unittest.TestCase):
    # README: too_broad는 HIGH 2 + REVIEW 1 = 3x2 + 1 = 점수 7
    def test_too_broad(self):
        f = scan(P["too_broad"])
        self.assertEqual(sorted(x["level"] for x in f), ["HIGH", "HIGH", "REVIEW"])
        self.assertEqual(risk_score(f), 7)

    # README: half_broad는 HIGH 1, REVIEW 5 = 3 + 5 = 점수 8 (점수가 높아도 HIGH 개수가 더 적을 수 있음)
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
    # README의 scoped 세 줄과 같은 질의: ALLOW, DENY(기본), DENY(명시)
    def test_default_deny_and_explicit_deny(self):
        s = P["scoped"]
        self.assertEqual(allowed(s, "s3:GetObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf"), "ALLOW")
        self.assertEqual(allowed(s, "s3:GetObject", "arn:aws:s3:::example-model-bucket/other/secret.txt"), "DENY(기본)")
        self.assertEqual(allowed(s, "s3:DeleteObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf"), "DENY(명시)")

    # Allow가 s3:* 전체여도 Deny가 적힌 DeleteObject는 거부, 나머지는 허용
    def test_explicit_deny_beats_allow(self):
        pol = {"Statement": [
            {"Effect": "Allow", "Action": "s3:*", "Resource": "*"},
            {"Effect": "Deny", "Action": "s3:DeleteObject", "Resource": "*"}]}
        self.assertEqual(allowed(pol, "s3:DeleteObject", "arn:aws:s3:::b/k"), "DENY(명시)")
        self.assertEqual(allowed(pol, "s3:GetObject", "arn:aws:s3:::b/k"), "ALLOW")

    def test_action_is_case_insensitive(self):
        pol = {"Statement": [{"Effect": "Allow", "Action": "IAM:listaccesskeys", "Resource": "*"}]}
        self.assertEqual(allowed(pol, "iam:ListAccessKeys", "arn:aws:iam::1:user/x"), "ALLOW")

    # 공식 문서의 와일드카드 예 패턴: *가 /를 넘어 여러 단계, 빈 문자열, 연속 슬래시와도 맞는지 확인(ok 목록)하고 맞지 않는 경우(no 목록)를 확인
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
