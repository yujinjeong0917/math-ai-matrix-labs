import unittest

from condition_ledger import COVERED, EXPERIMENTS, FIELDS, SHOP_SCREEN, coverage, differing


class Ledger(unittest.TestCase):
    def test_exp1_vs_exp2(self):
        d, u = differing(1, 2)
        self.assertEqual(len(d), 7)
        self.assertNotIn("모델", d)  # 모델은 둘 다 로지스틱 회귀
        self.assertEqual(u, [])

    def test_exp1_vs_exp3(self):
        d, u = differing(1, 3)
        self.assertEqual(d, ["모델", "모델 정확도(%)", "모델 AUC"])
        self.assertEqual(u, ["훈련 과제 수", "기본 보수(달러)"])

    def test_shop_screen(self):
        hit, miss = coverage(SHOP_SCREEN)
        self.assertEqual(hit, ["재는 결과"])
        self.assertEqual(len(miss), 6)

    def test_same_conditions_all_covered(self):
        deploy = {"과제": "재범 예측", "모델": "신경망", "출력 형식": "두 범주 예측과 확률", "설명 형식": "SHAP 기여도",
                  "참가자": "미국 거주자(온라인 작업 플랫폼)",
                  "과제 구성": "같은 과제 32건을 연달아, 건별 정답 피드백 없음", "재는 결과": "과신"}
        hit, miss = coverage(deploy)
        self.assertEqual(miss, [])
        self.assertEqual(len(hit), len(COVERED))

    def test_fields(self):
        for e in EXPERIMENTS.values():
            self.assertEqual(list(e), FIELDS)


if __name__ == "__main__":
    unittest.main()
