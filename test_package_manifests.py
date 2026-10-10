#!/usr/bin/env python3
"""Keep deployment-target manifests on the same public capability surface."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent
MANIFESTS = (
    "package_manifest.yaml",
    "package_manifest.jetson-native.yaml",
    "package_manifest.jetson-docker.yaml",
)
CAPABILITY = re.compile(r"^  - name: (robonix/\S+)$", re.MULTILINE)


def capabilities(manifest: str) -> list[str]:
    return CAPABILITY.findall((ROOT / manifest).read_text(encoding="utf-8"))


class PackageManifestTests(unittest.TestCase):
    def test_candidate_contract_is_local_and_referenced_by_every_target(self) -> None:
        contract = ROOT / "capabilities/get_pose_candidates.v1.toml"
        idl = ROOT / "capabilities/lib/map/srv/GetPoseCandidates.srv"
        self.assertTrue(contract.is_file())
        self.assertTrue(idl.is_file())
        self.assertIn("map/srv/GetPoseCandidates.srv", contract.read_text())
        for manifest in MANIFESTS:
            with self.subTest(manifest=manifest):
                text = (ROOT / manifest).read_text()
                self.assertIn(
                    "  - name: robonix/service/map/get_pose_candidates\n"
                    "    path: capabilities/get_pose_candidates.v1.toml",
                    text,
                )

    def test_all_targets_declare_the_same_capabilities(self) -> None:
        expected = capabilities(MANIFESTS[0])
        self.assertTrue(expected)
        self.assertNotIn("robonix/service/map/driver", expected)
        self.assertNotIn("robonix/lifecycle/driver", expected)
        for manifest in MANIFESTS[1:]:
            with self.subTest(manifest=manifest):
                self.assertEqual(expected, capabilities(manifest))


if __name__ == "__main__":
    unittest.main()
