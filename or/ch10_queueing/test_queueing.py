import math
import unittest
import queueing as q


class QueueingTest(unittest.TestCase):
    def test_formulas(self):
        self.assertAlmostEqual(q.mm1_mean_system(0.8), 25.0)
        self.assertAlmostEqual(q.mm1_mean_wait(0.8), 20.0)
        self.assertAlmostEqual(q.mm1_mean_system(0.9), 50.0)
        # rho 0.8: W = 5/0.2 = 25분, Wq = 20분(대기 = rho x 체류), 0.9에서는 50분
        # Little 법칙: L = lambda * W, Lq = rho^2/(1-rho)
        rho = 0.8
        self.assertAlmostEqual(q.lam_for(rho) * q.mm1_mean_wait(rho), rho ** 2 / (1 - rho))

    def test_tail_and_quantile_consistent(self):
        for rho in (0.5, 0.8, 0.9):
            # 95% 분위수 시간을 꼬리식에 다시 넣으면 정확히 5%가 남아야 한다(식끼리의 일관성)
            t = q.mm1_quantile(rho, 0.95)
            self.assertAlmostEqual(q.mm1_tail(rho, t), 0.05)
        self.assertAlmostEqual(q.mm1_tail(0.8, 30), math.exp(-1.2))

    def test_simulation_matches_formula(self):
        wq, w, ts = q.simulate(0.8)
        # 20만 건 시뮬레이션 평균이 공식(20, 25)에서 약 2% 이내 어긋난다. 20.51, 25.51은 seed=7에서 나온 값(난수 구현에 의존)
        self.assertAlmostEqual(wq, 20.51, places=2)   # seed=7 고정 결과
        self.assertAlmostEqual(w, 25.51, places=2)
        self.assertLess(abs(w - 25.0) / 25.0, 0.05)
        self.assertLess(abs(q.share_over(ts, 30) - q.mm1_tail(0.8, 30)), 0.02)

    def test_mean_ok_but_sla_fails(self):
        rho = 1 - q.MEAN_SERVICE / 30          # 평균 체류가 정확히 30분이 되는 이용률
        self.assertAlmostEqual(q.mm1_mean_system(rho), 30.0)
        self.assertGreater(q.mm1_tail(rho, 30), 0.35)   # 그래도 3명 중 1명 이상은 30분을 넘긴다

    def test_deterministic_service_halves_wait(self):
        # 처리 시간 변동이 없으면 평균 대기가 절반(20 -> 10). 시뮬레이션 10.14는 표본 오차 포함
        self.assertAlmostEqual(q.md1_mean_wait(0.8), q.mm1_mean_wait(0.8) / 2)
        wq, _, _ = q.simulate(0.8, deterministic_service=True)
        self.assertAlmostEqual(wq, 10.14, places=2)

    def test_nonlinear_growth(self):
        # 5/(1-0.972) = 178.571...: rho가 0.9 -> 0.972(8% 증가)일 때 평균 체류 50 -> 178.6분, 3.5배 초과
        self.assertAlmostEqual(q.mm1_mean_system(0.972), 178.5714, places=3)
        self.assertGreater(q.mm1_mean_system(0.972) / q.mm1_mean_system(0.9), 3.5)

    def test_deterministic_seed(self):
        self.assertEqual(q.simulate(0.5, n=1000)[0], q.simulate(0.5, n=1000)[0])


if __name__ == "__main__":
    unittest.main()
