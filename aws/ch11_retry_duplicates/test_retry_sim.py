import json
import unittest
from pathlib import Path

import retry_sim as r

P = json.loads(Path(__file__).with_name("retry_profile.json").read_text(encoding="utf-8"))


class RetrySimTest(unittest.TestCase):
    def test_batch_duplicates_with_default_visibility(self):
        b = r.batch_duplicates(P["batch_size"], P["seconds_per_request"], P["visibility_timeout_seconds"])
        self.assertEqual(b["done_at"][0], 25)
        self.assertEqual(b["done_at"][-1], 250)
        self.assertEqual(b["duplicate_count"], 9)
        self.assertEqual(b["duplicated_positions"], list(range(2, 11)))
        self.assertEqual(b["wasted_seconds"], 225)

    def test_long_enough_visibility_removes_duplicates(self):
        need = r.min_visibility_seconds(10, 40)
        self.assertEqual(need, 400)
        self.assertEqual(r.batch_duplicates(10, 25, 250)["duplicate_count"], 0)
        self.assertEqual(r.batch_duplicates(10, 25, 249)["duplicate_count"], 1)

    def test_batch_of_one_is_safe_when_fast_enough(self):
        self.assertEqual(r.batch_duplicates(1, 25, 30)["duplicate_count"], 0)

    def test_lambda_attempts(self):
        self.assertEqual(r.lambda_async_attempts(P["lambda_async_default_retries"]), 3)
        self.assertEqual(r.lambda_async_attempts(0), 1)

    def test_effects_with_and_without_idempotency(self):
        ids = [f"req-{i}" for i in range(1, 11)]
        delivered = ids + ids[:3]
        self.assertEqual(r.apply_effects(delivered, False), 13)
        self.assertEqual(r.apply_effects(delivered, True), 10)

    def test_fifo_window(self):
        sends = [(0, "a"), (120, "a"), (299, "a"), (300, "a"), (310, "b")]
        kept = r.fifo_accepted(sends, 300)
        self.assertEqual(kept, [(0, "a"), (300, "a"), (310, "b")])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            r.batch_duplicates(0, 25, 30)


if __name__ == "__main__":
    unittest.main()
