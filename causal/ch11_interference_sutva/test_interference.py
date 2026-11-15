"""11장 테스트: 그래프 모양(평균 이웃 수, 고립 노드, 커뮤니티 간 간선 비율)과 간섭 시나리오별 추정을 확인한다.
진짜 전체 출시 효과는 tau + gamma = 5(간섭 없음), 9(이웃에게 긍정적으로 번짐), 1(경쟁·잠식으로 줄어듦)이다."""
import unittest
import interference as f


class Interference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = f.report()

    def test_graph_shape(self):
        # 기대 평균 이웃 수: 49 x 0.20 + 950 x 0.002 = 9.8 + 1.9 = 11.7. 커뮤니티 밖 간선 비율은 기대 약 16%(1.9/11.7), 이 시드의 실측은 약 0.157
        self.assertAlmostEqual(self.r["mean_degree"], 11.7, delta=0.3)
        self.assertEqual(self.r["isolated"], 0)
        self.assertAlmostEqual(self.r["cross_edge_share"], 0.157, delta=0.01)

    def test_no_spillover_individual_ab_is_right(self):
        x = self.r["no_spillover"]
        self.assertAlmostEqual(x["truth"], 5.0, places=6)
        self.assertAlmostEqual(x["individual"], 5.0, delta=0.15)  # 배정 잡음 허용폭

    def test_positive_spillover_individual_ab_misses_it(self):
        x = self.r["spillover_pos"]
        self.assertAlmostEqual(x["truth"], 9.0, places=6)
        # 개인 A/B는 gamma 몫이 상쇄돼 여전히 약 5. 진짜 9를 4만큼 놓친다
        self.assertAlmostEqual(x["individual"], 5.0, delta=0.15)
        self.assertLess(abs(x["cluster"] - 9.0), abs(x["individual"] - 9.0))

    def test_negative_spillover_individual_ab_overstates(self):
        x = self.r["spillover_neg"]
        self.assertAlmostEqual(x["truth"], 1.0, places=6)
        # 잠식(gamma < 0)이 있으면 개인 A/B는 5로 효과를 5배 부풀린다
        self.assertAlmostEqual(x["individual"], 5.0, delta=0.15)
        self.assertLess(abs(x["cluster"] - 1.0), abs(x["individual"] - 1.0))

    def test_cluster_design_is_not_exact(self):
        x = self.r["spillover_pos"]
        self.assertGreater(9.0 - x["cluster"], 0.3)  # 커뮤니티 사이 간선 때문에 덜 맞는다


if __name__ == "__main__":
    unittest.main()
