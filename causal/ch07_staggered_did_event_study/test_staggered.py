"""7장 테스트: 손계산 2x2(이미 처치된 지역을 대조로 쓰면 0), TWFE 계수와 음수 가중치의 위치,
깨끗한 비교(ATT(g,t))의 이벤트스터디, 대조 지역을 뺐을 때 부호 반전, 효과가 일정할 때의 정상 동작을 확인한다.
기대값은 시드 11 고정 결과의 반올림이고, 가중치 합 등 구조적 값은 손으로 따라갈 수 있다."""
import os
import sys
import unittest

# 어느 디렉터리에서 실행해도 같은 폴더의 staggered.py를 찾도록 경로를 앞에 넣는다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staggered as s


class HandExample(unittest.TestCase):
    def test_already_treated_region_as_control_gives_zero(self):
        # B의 2->3주 변화 10에서 이미 처치된 A의 같은 기간 변화 10을 빼면 0. B의 진짜 효과 +10이 사라진다
        self.assertEqual(s.hand_two_by_two(s.HAND_B, s.HAND_A, 1, 2), 0.0)

    def test_clean_comparison_gives_ten(self):
        # A의 1->2주 변화 10에서 아직 처치 전인 B의 변화 0을 빼면 10(진짜 효과)
        self.assertEqual(s.hand_two_by_two(s.HAND_A, s.HAND_B, 0, 1), 10.0)


class Panel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = s.make_panel()
        cls.beta, cls.wts = s.twfe(cls.rows)

    def test_truth_and_twfe(self):
        # 처치 칸 240 = A 8주 x 20곳 + B 4주 x 20곳. 진짜 평균 11.50인데 TWFE는 5.88로 절반 수준
        self.assertEqual(sum(r["d"] for r in self.rows), 240)
        self.assertEqual(round(s.true_average_effect(self.rows), 2), 11.50)
        self.assertEqual(round(self.beta, 2), 5.88)
        self.assertEqual(round(s.weighted_average_effect(self.wts), 2), 6.00)

    def test_negative_weights_sit_on_already_treated_region(self):
        # 음수 가중치 80칸 = 이미 처치된 A의 7~10주차(4주 x 20곳). 합 -0.125
        by, neg, n_neg, n = s.weight_summary(self.wts)
        self.assertEqual((round(by["A"], 3), round(by["B"], 3), n_neg, n, round(neg, 3)), (0.375, 0.625, 80, 240, -0.125))
        parts = {k: (round(w, 3), round(m, 2)) for k, w, m in s.weight_parts(self.wts)}
        self.assertEqual(parts, {"A 3~6주차": (0.5, 7.5), "A 7~10주차": (-0.125, 19.5), "B 7~10주차": (0.625, 7.5)})
        # 가중치 x 진짜 평균 효과 = 0.5x7.5 - 0.125x19.5 + 0.625x7.5 = 6.0. 효과가 가장 큰 칸(19.5)에 음수 가중치가 붙어 평균이 끌려 내려간다
        self.assertAlmostEqual(sum(w * m for _, w, m in s.weight_parts(self.wts)), 6.0, places=9)
        late_a = [w for r, w in self.wts if r["g"] == "A" and r["t"] >= s.GROUPS["B"]]
        self.assertEqual(len(late_a), 80)
        neg = [(r["g"], r["t"]) for r, w in self.wts if w < 0]
        self.assertEqual(set(neg), {("A", t) for t in range(7, 11)})
        self.assertTrue(all(w < 0 for w in late_a))
        self.assertAlmostEqual(abs(sum(w for _, w in self.wts) - 1.0), 0.0, places=9)  # 정규화로 가중치 합은 1

    def test_gap_between_6_and_5_88_is_noise(self):
        # 잡음 없이 구하면 정확히 6.00. 5.88과의 0.12 차이는 잡음(구조적 편향 11.5 -> 6.0과는 별개)
        quiet, _ = s.twfe(s.make_panel(noise=0.0))
        self.assertEqual(round(quiet, 2), 6.00)
        self.assertEqual(round(self.beta, 2), 5.88)

    def test_clean_estimator_recovers_event_study(self):
        att = s.att_gt(self.rows)
        es = s.event_study(att)
        # 키는 상대 시점 e(도입 주 = 0). e=-1은 기준 주라 없다. 처치 전(e<0)은 0 근처, 이후는 진짜 효과 3(e+1)에 가깝게 따라 올라간다
        # (e=4~7은 A만 남아 B와 섞이지 않는다)
        want = {-6: 0.14, -5: 0.35, -4: 1.21, -3: 0.59, -2: -0.88, 0: 2.93, 1: 5.78, 2: 8.88,
                3: 11.93, 4: 15.18, 5: 17.96, 6: 20.92, 7: 23.77}
        self.assertEqual({e: round(v, 2) for e, v in es.items()}, want)
        self.assertEqual(round(s.overall_att(att), 2), 11.41)
        self.assertTrue(all(abs(es[e]) < 1.5 for e in es if e < 0))  # 사전추세 검사: 가짜 효과의 허용폭 1.5(잡음 수준)

    def test_without_never_treated_region_sign_flips(self):
        # 끝까지 처치되지 않는 C를 빼면 B가 A의 이미 처치된 기간을 대조로 쓰게 되어 계수가 -0.66으로 부호가 뒤집힌다
        two = [r for r in self.rows if r["g"] != "C"]
        beta, w = s.twfe(two)
        self.assertEqual(round(s.true_average_effect(two), 2), 11.50)
        self.assertEqual(round(beta, 2), -0.66)
        self.assertEqual(round(s.weighted_average_effect(w), 2), -0.50)
        parts = {k: (round(a, 3), round(m, 2)) for k, a, m in s.weight_parts(w)}
        self.assertEqual(parts, {"A 3~6주차": (1.0, 7.5), "A 7~10주차": (-0.667, 19.5), "B 7~10주차": (0.667, 7.5)})
        self.assertEqual(sum(1 for _, x in w if x < 0), 80)

    def test_constant_effect_is_fine(self):
        # 효과를 +8 일정으로 덮어쓰면 음수 가중치가 있어도 모든 칸의 효과가 같아 TWFE가 약 8을 되찾는다(7.88은 잡음)
        flat = s.make_panel()
        for r in flat:
            if r["d"] == 1:
                r["y"] += 8.0 - r["tau"]
                r["tau"] = 8.0
        self.assertEqual(round(s.twfe(flat)[0], 2), 7.88)
        quiet = s.make_panel(noise=0.0)
        for r in quiet:
            if r["d"] == 1:
                r["y"] += 8.0 - r["tau"]
                r["tau"] = 8.0
        self.assertEqual(round(s.twfe(quiet)[0], 2), 8.00)


if __name__ == "__main__":
    unittest.main()
