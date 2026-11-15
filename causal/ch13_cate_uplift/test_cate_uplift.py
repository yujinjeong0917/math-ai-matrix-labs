"""13장 테스트: 시드 고정 재현성, 전체 평균효과 추정의 오차 범위, 사전 지정 하위집단의 효과 차이,
업리프트 정책이 모두에게 주기보다 낫다는 점, 효과 0 세계에서 사후 하위집단 탐색이 만드는 가짜 유의성과 그 보정을 확인한다."""
import unittest

from cate_uplift import (COST, diff_means, make_customers, policy_gain, subgroup_experiment,
                         t_learner_cells, treat_all_gain, true_tau, z_for_two_sided)


class Ch13(unittest.TestCase):
    def setUp(self):
        self.rows = make_customers(20000, seed=2)
        half = len(self.rows) // 2
        self.train, self.hold = self.rows[:half], self.rows[half:]

    def test_deterministic(self):
        self.assertEqual(diff_means(self.rows), diff_means(make_customers(20000, seed=2)))

    def test_ate_within_three_se(self):
        est, se = diff_means(self.rows)
        truth = sum(true_tau(r["new"], r["heavy"]) for r in self.rows) / len(self.rows)
        self.assertLess(abs(est - truth), 3 * se)  # 추정이 진짜 평균에서 표준오차 3배 이내(우연히 어긋날 확률이 아주 작다)

    def test_prespecified_segments_differ(self):
        new, _ = diff_means([r for r in self.rows if r["new"] == 1])
        old, _ = diff_means([r for r in self.rows if r["new"] == 0])
        # 진짜 평균 효과는 신규 약 3050, 기존 약 550(격차 약 2500)이라 2000을 넘는다. 사전에 정한 하위집단이라 다중비교 부담이 없다
        self.assertGreater(new - old, 2000)

    def test_uplift_policy_beats_treat_all(self):
        cells = t_learner_cells(self.train)
        # 학습 절반으로 칸 추정을 만들고 보류 절반으로 평가한다. 기존+다량 같은 손해 칸을 거르는 몫이 300원/명 넘게 이득
        self.assertGreater(policy_gain(cells, self.hold), treat_all_gain(self.hold) + 300)
        self.assertGreater(policy_gain(cells, self.hold), 0)

    def test_cost_value(self):
        self.assertEqual(COST, 1500)

    def test_post_hoc_subgroup_trap(self):
        # 효과가 전부 0인 세계에서 40개 하위집단 중 최선을 고르면 보정 없이는 40% 넘게 "유의"가 나온다(표준 기대는 5%)
        res = subgroup_experiment()
        self.assertAlmostEqual(res["z05"], 1.96, places=2)
        self.assertGreater(res["share_best_sig"], 0.4)
        self.assertLess(res["share_best_sig_bonf"], 0.10)   # 본페로니 보정 후에는 5% 안팎(0.10 미만)
        self.assertLess(abs(res["holdout_mean_est"]), 100)  # 보류 데이터에서는 효과가 0 근처로 돌아온다(단위: 원)
        self.assertLess(res["holdout_share_sig"], 0.12)     # 보류 검정의 가짜 유의 비율도 약 5%(0.12 미만)

    def test_z_inverse(self):
        self.assertAlmostEqual(z_for_two_sided(0.05), 1.96, places=2)


if __name__ == "__main__":
    unittest.main()
