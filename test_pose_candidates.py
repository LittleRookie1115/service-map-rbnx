# SPDX-License-Identifier: MulanPSL-2.0
"""Candidate gating and operator seed handover without a live ROS graph."""
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mapping_rbnx import global_match, map_ops


class PoseCandidateTests(unittest.TestCase):
    def _query(self, best, second, live=None):
        current = live or {"map_id": "lab", "mode": "localization", "generation": 7}
        match = global_match.Match(
            (1.0, 2.0, 0.3), best, second, "test scan",
            [global_match.Candidate((1.0, 2.0, 0.3), best),
             global_match.Candidate((5.0, 6.0, 1.2), second)])
        with (
            patch.object(map_ops, "active_algo", return_value="slam_toolbox"),
            patch.object(map_ops.lifecycle, "current", return_value=current),
            patch.object(map_ops.localizers, "relocalization", return_value={"state": "running"}),
            patch.object(map_ops.localizers, "load_grid", return_value=object()),
            patch.object(map_ops.localizers, "match_statically", return_value=(match, "")),
            patch.object(map_ops, "_get_node", return_value=object()),
        ):
            return map_ops.get_pose_candidates_impl()

    def test_clear_match_does_not_offer_manual_choice(self):
        result = self._query(0.90, 0.60)
        self.assertTrue(result["ok"])
        self.assertFalse(result["needs_selection"])
        self.assertEqual(result["candidates_json"], "[]")

    def test_low_score_or_close_runner_up_offers_ranked_top_k(self):
        for best, second in ((0.70, 0.40), (0.90, 0.85)):
            with self.subTest(best=best, second=second):
                result = self._query(best, second)
                self.assertTrue(result["needs_selection"])
                self.assertEqual((result["map_id"], result["generation"]), ("lab", 7))
                poses = json.loads(result["candidates_json"])
                self.assertEqual([p["rank"] for p in poses], [1, 2])
                self.assertEqual((poses[1]["x"], poses[1]["theta"]), (5.0, 1.2))

    def test_confirmed_localization_does_not_query_candidates(self):
        with (
            patch.object(map_ops, "active_algo", return_value="slam_toolbox"),
            patch.object(map_ops.lifecycle, "current", return_value={
                "map_id": "lab", "mode": "localization", "generation": 7}),
            patch.object(map_ops.localizers, "relocalization", return_value={"state": "done"}),
            patch.object(map_ops.localizers, "match_statically") as matcher,
        ):
            result = map_ops.get_pose_candidates_impl()
        self.assertTrue(result["ok"])
        self.assertFalse(result["needs_selection"])
        matcher.assert_not_called()

    def test_changed_map_discards_candidate_result(self):
        first = {"map_id": "lab", "mode": "localization", "generation": 7}
        second = {**first, "generation": 8}
        with (
            patch.object(map_ops, "active_algo", return_value="slam_toolbox"),
            patch.object(map_ops.lifecycle, "current", side_effect=[first, second]),
            patch.object(map_ops.localizers, "relocalization", return_value={"state": "running"}),
            patch.object(map_ops.localizers, "load_grid", return_value=object()),
            patch.object(map_ops.localizers, "match_statically", return_value=(
                global_match.Match((1, 2, 0), 0.6, 0.55, "scan"), "")),
            patch.object(map_ops, "_get_node", return_value=object()),
        ):
            result = map_ops.get_pose_candidates_impl()
        self.assertFalse(result["ok"])
        self.assertEqual(result["candidates_json"], "[]")

    def test_non_slam_toolbox_engine_rejects_candidate_query(self):
        with (
            patch.object(map_ops, "active_algo", return_value="rtabmap"),
            patch.object(map_ops.localizers, "match_statically") as matcher,
        ):
            result = map_ops.get_pose_candidates_impl()
        self.assertFalse(result["ok"])
        self.assertFalse(result["needs_selection"])
        self.assertIn("slam_toolbox", result["detail"])
        matcher.assert_not_called()

    def test_operator_seed_joins_pending_relocalization(self):
        with (
            patch.object(map_ops, "active_algo", return_value="slam_toolbox"),
            patch.object(map_ops, "_get_node", return_value=object()),
            patch.object(map_ops, "_RELOC_ACCEPTING_SEED", True),
        ):
            result = map_ops.pose_estimate_impl(1.0, 2.0, 0.3)
            self.assertTrue(result["ok"])
            self.assertEqual(map_ops._take_relocalization_seed(), (1.0, 2.0, 0.3))
        self.assertFalse(map_ops._RELOC_SEED_EVENT.is_set())

    def test_rtabmap_pose_estimate_keeps_the_initialpose_path(self):
        class Message:
            def __init__(self):
                self.header = SimpleNamespace(frame_id="", stamp=None)
                self.pose = SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=0.0, y=0.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
                    ),
                    covariance=[],
                )

        published = []
        publisher = SimpleNamespace(publish=published.append)
        node = SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: object())),
            create_publisher=lambda *_args: publisher,
            destroy_publisher=lambda _publisher: None,
        )
        geometry_msgs = types.ModuleType("geometry_msgs")
        geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
        geometry_msgs_msg.PoseWithCovarianceStamped = Message
        geometry_msgs.msg = geometry_msgs_msg
        with (
            patch.dict(sys.modules, {
                "geometry_msgs": geometry_msgs,
                "geometry_msgs.msg": geometry_msgs_msg,
            }),
            patch.object(map_ops, "active_algo", return_value="rtabmap"),
            patch.object(map_ops, "_get_node", return_value=node),
            patch.object(map_ops, "_RELOC_ACCEPTING_SEED", True),
            patch.object(map_ops.time, "sleep"),
        ):
            result = map_ops.pose_estimate_impl(1.0, 2.0, 0.3)
        self.assertTrue(result["ok"])
        self.assertIn(map_ops.INITIALPOSE_TOPIC, result["detail"])
        self.assertEqual(len(published), 1)

    def test_selected_pose_requires_a_live_scan_check(self):
        with patch.object(map_ops.localizers, "scan_fit", return_value=(-1.0, "no fresh scan")):
            ok, detail = map_ops._confirm_against_map(object(), object(), (1, 2, 0),
                                                       require_scan=True)
        self.assertFalse(ok)
        self.assertIn("no fresh scan", detail)


if __name__ == "__main__":
    unittest.main()
