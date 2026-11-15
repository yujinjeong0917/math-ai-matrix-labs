"""4장 테스트: 손계산 표(IPW가 효과 8을 복원), 겹침이 충분한 장면(IPW·매칭이 진짜 효과에 근접),
겹침이 부족한 장면(가중치 폭발, 편향), 200회 반복 요약을 확인한다.
기대값은 모두 고정 시드(2000명, seed 7 등)에서 나온 값을 반올림한 것이다."""
import os
import sys
import unittest

# 어느 디렉터리에서 실행해도 같은 폴더의 propensity.py를 찾도록 경로를 앞에 넣는다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import propensity as p

# 겹침 부족 설정: 성향점수 = sigmoid(-5 + 4x)라 x < 0인 고객은 거의 쿠폰을 못 받고, 효과는 8 + 4x로 x에 따라 커진다
KW_POOR = dict(slope=4.0, intercept=-5.0, tau0=8.0, tau1=4.0)


class HandTable(unittest.TestCase):
    def test_naive_gap_is_much_larger_than_effect(self):
        # 처치군 평균 63.6 = (80x70 + 20x38)/100, 비처치군 36.4 = (20x62 + 80x30)/100, 차이 27.2는 진짜 효과 8의 3배 넘게 부풀려진 값
        m1, m0, d = p.hand_naive()
        self.assertEqual((round(m1, 1), round(m0, 1), round(d, 1)), (63.6, 36.4, 27.2))

    def test_stratified_and_ipw_recover_eight(self):
        self.assertAlmostEqual(p.hand_stratified(), 8.0)  # 두 층 모두 효과 8이라 평균도 8
        # IPW의 54.0과 46.0은 전체 모집단의 평균 y1((70+38)/2)과 y0((62+30)/2)을 복원한 값
        a, b, c = p.hand_ipw()
        self.assertEqual((round(a, 1), round(b, 1), round(c, 1)), (54.0, 46.0, 8.0))


class GoodOverlap(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_data(2000, 7)
        self.ps, self.beta = p.propensity_scores(self.rows)

    def test_data_is_deterministic(self):
        self.assertEqual(sum(r["t"] for r in self.rows), 940)  # 시드 고정이라 처치 인원이 항상 같다

    def test_logit_coefficients(self):
        # 진짜 계수(절편 -0.2, 기울기 1.0)에 가까운 추정값이 나오는지 확인
        self.assertEqual((round(self.beta[0], 2), round(self.beta[1], 2)), (-0.17, 0.95))

    def test_naive_is_biased_ipw_and_matching_are_close(self):
        self.assertEqual(round(p.naive_diff(self.rows), 2), 15.67)
        self.assertEqual(round(p.ipw_ate(self.rows, self.ps), 2), 7.97)
        att, n_t, n_c, _ = p.match_att(self.rows, self.ps)
        # 처치 940명이 서로 다른 대조 465명을 짝으로 쓴다(복원 추출이라 같은 대조가 여러 번 재사용)
        self.assertEqual((round(att, 2), n_t, n_c), (7.93, 940, 465))

    def test_weighting_balances_x(self):
        w = p.weights(self.rows, self.ps)
        # 표준화 평균 차이가 0.87 -> 0.03으로 줄면 가중 후 두 집단의 x 분포가 맞춰진 것
        self.assertEqual(round(p.smd(self.rows), 2), 0.87)
        self.assertEqual(round(p.smd(self.rows, w), 2), 0.03)
        self.assertEqual(round(max(w), 1), 12.6)
        self.assertEqual(round(p.effective_sample_size(w)), 1599)  # 2000명 중 약 1600명분의 정보가 남는다


class PoorOverlap(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_data(2000, 7, **KW_POOR)
        self.ps, _ = p.propensity_scores(self.rows)

    def test_ipw_ate_is_far_from_truth(self):
        self.assertEqual(round(p.true_ate(self.rows), 2), 8.11)
        # 진짜 ATE 8.11인데 IPW는 19.52: 처치군이 거의 없는 x < 0 구간을 소수의 처치군 가중치로 메우려다 크게 어긋난다
        self.assertEqual(round(p.ipw_ate(self.rows, self.ps), 2), 19.52)

    def test_matching_att_and_trimming(self):
        # ATT 14.01 > ATE 8.11: 효과가 x와 함께 커지고 처치군이 x가 큰 쪽에 몰려서
        self.assertEqual(round(p.true_att(self.rows), 2), 14.01)
        att, _, n_c, _ = p.match_att(self.rows, self.ps)
        self.assertEqual((round(att, 2), n_c), (15.19, 106))
        # 성향점수 0.1~0.9만 남기면 436명. 추정 대상이 그 집단의 효과(12.43)로 바뀌므로 비교 기준도 tau다
        est, n, tau = p.trimmed_ipw(self.rows, self.ps)
        self.assertEqual((round(est, 2), n, round(tau, 2)), (11.18, 436, 12.43))

    def test_almost_no_treated_below_zero(self):
        # x < 0 고객 985명 중 쿠폰을 받은 사람이 1명뿐: 겹침 부족의 직접 증거
        low = [r for r in self.rows if r["x"] < 0]
        self.assertEqual((len(low), sum(r["t"] for r in low)), (985, 1))

    def test_treated_side_overrepresents_high_activity(self):
        w = p.weights(self.rows, self.ps)
        wt = [(r, wi) for r, wi in zip(self.rows, w) if r["t"] == 1]
        wc = [(r, wi) for r, wi in zip(self.rows, w) if r["t"] == 0]
        # 가중 평균 x: 쿠폰 쪽 0.72, 못 받은 쪽 0.02, 전체 0.03. 쿠폰 쪽 가중 평균이 전체와 크게 어긋나 대표성이 깨진다
        self.assertEqual((round(p.weighted_mean_x(wt), 2), round(p.weighted_mean_x(wc), 2)), (0.72, 0.02))
        self.assertEqual(round(p.mean([r["x"] for r in self.rows]), 2), 0.03)
        # 가장 큰 가중치 한 명(x = -0.14)이 쿠폰 쪽 가중치 합의 21%를 차지한다
        top = max(wt, key=lambda a: a[1])
        self.assertEqual((round(top[0]["x"], 2), round(top[1] / sum(wi for _, wi in wt), 2)), (-0.14, 0.21))

    def test_weights_explode(self):
        w = p.weights(self.rows, self.ps)
        self.assertEqual(round(max(w), 1), 232.8)  # 충분한 겹침 장면의 12.6과 대비
        self.assertEqual(round(p.effective_sample_size(w)), 139)  # 2000명이 사실상 139명분


class Replication(unittest.TestCase):
    def test_summary_over_200_runs(self):
        # 각 설정에서 200번 반복한 IPW의 평균 오차(편향)와 표준편차. 겹침 부족은 편향 7.18, 표준편차 4.16으로 모두 크다
        for kw, bias, spread in (({}, 0.04, 0.38), (KW_POOR, 7.18, 4.16)):
            res = p.replicate(200, 2000, 100, **kw)
            err = [a[1] - a[2] for a in res]
            self.assertEqual(round(p.mean(err), 2), bias)
            self.assertEqual(round(p.sd([a[1] for a in res]), 2), spread)


if __name__ == "__main__":
    unittest.main()
