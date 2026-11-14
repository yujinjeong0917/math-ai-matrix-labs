import copy
import json
import unittest
from pathlib import Path

from artifact_rules import (MIB, check_release, min_part_bytes, multipart_plan,
                            stored_bytes, verdict)

MANIFEST = json.loads((Path(__file__).parent / "release_manifest.json").read_text(encoding="utf-8"))


def good():
    m = copy.deepcopy(MANIFEST)
    m["image"]["reference"] = m["image"]["repository"] + "@sha256:" + "ab" * 32
    m["code_commit"] = "3d8cd4c"
    m["bucket"]["versioning"] = "Enabled"
    m["repository"]["tag_mutability"] = "IMMUTABLE"
    return m


class ReleaseTest(unittest.TestCase):
    def test_sample_is_blocked_with_five_fails_one_warn(self):
        f = check_release(MANIFEST)
        self.assertEqual([x["rule"] for x in f], ["R3", "R4", "R5", "R6", "R7"])
        self.assertEqual(sum(x["level"] == "FAIL" for x in f), 4)
        self.assertEqual(verdict(MANIFEST), "BLOCK")

    def test_good_release_passes(self):
        self.assertEqual(check_release(good()), [])
        self.assertEqual(verdict(good()), "PASS")

    def test_warn_only(self):
        m = good()
        m["repository"]["tag_mutability"] = "MUTABLE"
        self.assertEqual(verdict(m), "WARN")

    def test_missing_version_id_and_bad_sha(self):
        m = good()
        m["model"]["s3_version_id"] = ""
        m["model"]["sha256"] = "xyz"
        self.assertEqual([x["rule"] for x in check_release(m)], ["R1", "R2"])


class StorageTest(unittest.TestCase):
    def test_versions_store_whole_objects(self):
        self.assertEqual(stored_bytes(4_683_000_000, 10), 46_830_000_000)
        self.assertEqual(stored_bytes(4_683_000_000, 1), 4_683_000_000)

    def test_multipart(self):
        self.assertEqual(multipart_plan(4_683_000_000, 100 * MIB), {"parts": 45, "fits": True})
        self.assertEqual(min_part_bytes(4_683_000_000), 5 * MIB)
        self.assertEqual(multipart_plan(4_683_000_000, 5 * MIB)["parts"], 894)

    def test_huge_object_needs_bigger_parts(self):
        size = 40 * 1024**4  # 40 TiB는 5MiB 파트로는 10,000개를 넘는다
        self.assertFalse(multipart_plan(size, 5 * MIB)["fits"])

    def test_bad_part_size(self):
        with self.assertRaises(ValueError):
            multipart_plan(10, 1)


if __name__ == "__main__":
    unittest.main()
