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
        for est in (before_after_treated(self.rows), naive_post_diff(self.rows), did_estimate(self.rows)):
            self.assertGreater(est - TRUE_EFFECT, 300)

    def test_structure_matched_method_recovers(self):
        for bw in (5, 10, 20):
            est, n, se = rdd_local_linear(self.rows, bw)
            self.assertLess(abs(est - TRUE_EFFECT), 3 * se)
            self.assertGreater(n, 1000)

    def test_cutoff_map(self):
        self.assertEqual(chapters(dict(assignment="cutoff"))[1], [8, 14])

    def test_random_noncompliance(self):
        self.assertEqual(chapters(dict(assignment="random", full_compliance=False))[1], [2, 9, 14])

    def test_staggered(self):
        self.assertEqual(chapters(dict(assignment="staggered"))[1][0], 7)

    def test_one_unit_needs_pool(self):
        self.assertEqual(chapters(dict(assignment="one_unit", pre_period=True, untreated_group=False)), ([10], []))
        self.assertEqual(chapters(dict(assignment="one_unit", pre_period=True, untreated_group=True))[1][0], 10)

    def test_self_selected_without_anything_is_blocked(self):
        self.assertEqual(chapters(dict(assignment="self_selected")), ([3], []))

    def test_self_selected_did_and_adjustment(self):
        _, p = chapters(dict(assignment="self_selected", pre_period=True, untreated_group=True,
                             covariates_cover_selection=True))
        self.assertEqual(p, [6, 3, 4, 5, 14])

    def test_spillover_blocks_first(self):
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
