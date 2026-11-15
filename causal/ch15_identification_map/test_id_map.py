"""15장 테스트: 이름이 안 맞는 방법의 편향, 구조에 맞는 회귀불연속의 복원, recommend 규칙의 출력(후보 장 번호 목록과 순서)을 확인한다.
chapters()는 recommend의 결과에서 장 번호만 뽑는 보조 함수다."""
import unittest

from id_map import (CHAPTERS, TRUE_EFFECT, before_after_treated, did_estimate, make_cutoff_data,
                    naive_post_diff, rdd_local_linear, recommend)


def chapters(f):
    b, p = recommend(f)
    return [c for c, _ in b], [c for c, _ in p]


class Ch15(unittest.TestCase):
    def setUp(self):
        self.rows = make_cutoff_data()

    def test_wrong_names_are_biased(self):
        # 세 방법 모두 진짜 1500보다 300원 넘게 크다(점수가 높을수록 성장이 빠른 구조를 못 거른다)
        for est in (before_after_treated(self.rows), naive_post_diff(self.rows), did_estimate(self.rows)):
            self.assertGreater(est - TRUE_EFFECT, 300)

    def test_structure_matched_method_recovers(self):
        for bw in (5, 10, 20):
            est, n, se = rdd_local_linear(self.rows, bw)
            self.assertLess(abs(est - TRUE_EFFECT), 3 * se)  # 진짜 효과가 추정 표준오차 3배 안
            self.assertGreater(n, 1000)

    def test_cutoff_map(self):
        # 기준선 배정 -> 8장(회귀불연속), 마지막은 항상 14장(강건성)
        self.assertEqual(chapters(dict(assignment="cutoff"))[1], [8, 14])

    def test_random_noncompliance(self):
        self.assertEqual(chapters(dict(assignment="random", full_compliance=False))[1], [2, 9, 14])

    def test_staggered(self):
        self.assertEqual(chapters(dict(assignment="staggered"))[1][0], 7)

    def test_one_unit_needs_pool(self):
        # 처치 단위 1개인데 비교 풀이 없으면 후보 없이 10장이 "막힘"으로만 나온다(후보가 비어 14장도 안 붙는다)
        self.assertEqual(chapters(dict(assignment="one_unit", pre_period=True, untreated_group=False)), ([10], []))
        self.assertEqual(chapters(dict(assignment="one_unit", pre_period=True, untreated_group=True))[1][0], 10)

    def test_self_selected_without_anything_is_blocked(self):
        # 자기선택인데 도구, 전후 비교, 관측 통제 모두 없으면 막힘 3장 하나뿐이고 후보가 없다
        self.assertEqual(chapters(dict(assignment="self_selected")), ([3], []))

    def test_self_selected_did_and_adjustment(self):
        # 순서: 6(이중차분), 3(DAG), 4(성향점수), 5(이중강건), 14(강건성)
        _, p = chapters(dict(assignment="self_selected", pre_period=True, untreated_group=True,
                             covariates_cover_selection=True))
        self.assertEqual(p, [6, 3, 4, 5, 14])

    def test_spillover_blocks_first(self):
        # 간섭이 있으면 무작위 배정이어도 11장이 막힘 목록에 먼저 올라온다
        b, _ = chapters(dict(assignment="random", spillover=True))
        self.assertEqual(b, [11])

    def test_heterogeneity_adds_13_before_14(self):
        _, p = chapters(dict(assignment="cutoff", effect_varies_question=True))
        self.assertEqual(p, [8, 13, 14])

    def test_every_pick_has_chapter_title(self):
        for c in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            self.assertIn(c, CHAPTERS)


if __name__ == "__main__":
    unittest.main()
