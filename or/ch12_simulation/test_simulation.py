import unittest
import simulation as sim

SEEDS = list(range(1, 21))


class SimulationTest(unittest.TestCase):
    def test_engine_matches_mm1_theory(self):
        got, theory = sim.mm1_check()
        # 이론 Wq = 0.8 x 5 / 0.2 = 20분. 시뮬레이션은 3% 이내면 엔진이 맞다고 본다(README 19.71)
        self.assertAlmostEqual(theory, 20.0)
        self.assertLess(abs(got - theory) / theory, 0.03)

    def test_same_seed_same_result(self):
        # 같은 시드는 같은 결과, 다른 시드는 다른 결과(난수를 모두 Random(시드)에서 뽑기 때문)
        self.assertEqual(sim.one_replication(8, 2), sim.one_replication(8, 2))
        self.assertNotEqual(sim.one_replication(8, 2), sim.one_replication(9, 2))

    def test_single_runs_differ_a_lot(self):
        xs = sim.replicate(2, SEEDS)
        # 상담원 2명으로 한 번만 돌리면 시드에 따라 0.62분부터 13.98분까지(10배 이상): 한 번의 실행으로 결론 내리면 안 된다
        self.assertAlmostEqual(min(xs), 0.62, places=2)
        self.assertAlmostEqual(max(xs), 13.98, places=2)
        self.assertGreater(max(xs) / min(xs), 10)

    def test_confidence_interval(self):
        m, lo, hi = sim.ci95(sim.replicate(2, SEEDS))
        # 20회 평균 3.77분, 95% 구간 2.20~5.34(t 구간이라 한쪽으로 치우친 값에는 근사)
        self.assertAlmostEqual(m, 3.77, places=2)
        self.assertAlmostEqual(lo, 2.20, places=2)
        self.assertAlmostEqual(hi, 5.34, places=2)

    def test_extra_agent_cuts_wait(self):
        m3, lo3, hi3 = sim.ci95(sim.replicate(3, SEEDS))
        self.assertAlmostEqual(m3, 0.58, places=2)
        # 3명 구간의 상한이 2명 구간의 하한(2.20)보다 작다
        self.assertLess(hi3, 2.20)             # 두 구간이 겹치지 않는다

    def test_common_random_numbers(self):
        # 같은 시드로 짝지은 차이(공통 난수) 대 시드를 따로 쓴 차이(101~120): 짝지으면 구간 폭이 좁다
        d = [a - b for a, b in zip(sim.replicate(2, SEEDS), sim.replicate(3, SEEDS))]
        m, lo, hi = sim.ci95(d)
        self.assertAlmostEqual(m, 3.19, places=2)
        d2 = [a - b for a, b in zip(sim.replicate(2, SEEDS), sim.replicate(3, range(101, 121)))]
        m2, lo2, hi2 = sim.ci95(d2)
        self.assertLess(hi - lo, hi2 - lo2)    # 같은 난수로 짝지으면 구간이 좁다

    def test_peak_hours_hidden_by_daily_mean(self):
        # 바쁜 시간대(처음 120분)에 도착한 고객의 평균 대기가 하루 평균보다 크다
        pk = []
        for s in SEEDS:
            arr = sim.make_arrivals(s)
            w = sim.run(arr, sim.make_services(s, len(arr)), 2)
            pk.append(sum(x for a, x in zip(arr, w) if a < sim.PEAK_END) / sum(1 for a in arr if a < sim.PEAK_END))
        self.assertGreater(sum(pk) / 20, sim.ci95(sim.replicate(2, SEEDS))[0])


if __name__ == "__main__":
    unittest.main()
