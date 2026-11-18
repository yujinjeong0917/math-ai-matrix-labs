import itertools
import unittest

from experiment_design import (assign_participants, balanced_latin_square, n_per_group,
                               ordered_pair_counts, position_counts, power_for_n, total_people)


class Design(unittest.TestCase):
    def test_square_for_4_matches_hand_table(self):
        self.assertEqual(balanced_latin_square(4), [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]])

    def test_position_balance(self):
        for n in (2, 4, 6):
            sq = balanced_latin_square(n)
            for pos in range(n):
                self.assertEqual(sorted(r[pos] for r in sq), list(range(n)))

    def test_pairs_each_once(self):
        for n in (4, 6):
            c = ordered_pair_counts(balanced_latin_square(n))
            self.assertEqual(set(c), set(itertools.permutations(range(n), 2)))
            self.assertEqual(set(c.values()), {1})

    def test_odd_rejected(self):
        with self.assertRaises(ValueError):
            balanced_latin_square(3)

    def test_assign_uses_all_rows(self):
        orders = assign_participants(8, list("abcd"), seed=7)
        self.assertEqual(len({tuple(o) for o in orders}), 4)

    def test_sample_size(self):
        self.assertEqual(n_per_group(0.4), 100)
        self.assertEqual(n_per_group(0.2), 394)
        self.assertEqual(n_per_group(0.5), 64)
        self.assertEqual(n_per_group(0.8), 26)
        self.assertEqual(n_per_group(0.85), 23)  # 1장의 극단적 대비 값을 넣으면 인원이 크게 줄어든다
        self.assertGreater(n_per_group(0.4, alpha=0.05 / 6), 100)

    def test_power(self):
        self.assertAlmostEqual(power_for_n(0.4, 100), 0.80, delta=0.02)
        self.assertLess(power_for_n(0.4, 50), 0.6)

    def test_excluded_share(self):
        from experiment_design import excluded_share
        self.assertAlmostEqual(excluded_share(63, 303), 20.79, places=2)

    def test_position_counts_balanced_only_when_multiple_of_4(self):
        eight = assign_participants(8, list("abcd"), seed=7)
        self.assertEqual(position_counts(eight, "a"), [2, 2, 2, 2])
        ten = assign_participants(10, list("abcd"), seed=7)
        self.assertNotEqual(position_counts(ten, "a"), [2, 2, 2, 2])

    def test_total_people(self):
        self.assertEqual(total_people(100, 4), 400)


if __name__ == "__main__":
    unittest.main()
