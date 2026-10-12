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
    def test_candidate_reads_are_cached_and_scoped_to_run(self):
        live = {"map_id": "lab", "generation": 7, "mode": "localization"}
        candidate = {"rank": 1, "x": 1, "y": 2, "theta": 0, "score": .8}
        for phase, generation, expected in [("choosing",7,True),("searching",7,False),("ready",7,False),("choosing",8,False)]:
            with (
                self.subTest(phase=phase, generation=generation),
                patch.object(map_ops, "active_algo", return_value="slam_toolbox"),
                patch.object(map_ops.lifecycle, "current", return_value=live),
                patch.object(map_ops._LOCALIZATION, "snapshot", return_value={
                    "map_id": "lab", "generation": generation, "phase": phase,
                    "candidates": [candidate], "detail": "current run"}),
                patch.object(map_ops.localizers, "match_statically") as matcher,
            ):
                for _ in range(3):
                    result = map_ops.get_pose_candidates_impl()
                    self.assertEqual(result["needs_selection"], expected)
                matcher.assert_not_called()

    def test_operator_seed_is_submitted_to_the_current_run(self):
        with (
            patch.object(map_ops, "active_algo", return_value="slam_toolbox"),
            patch.object(map_ops, "_get_node", return_value=object()),
            patch.object(map_ops._LOCALIZATION, "snapshot", return_value={"phase": "choosing", "map_id": "lab", "generation": 1}),
            patch.object(map_ops.lifecycle, "current", return_value={"map_id": "lab", "generation": 1}),
            patch.object(map_ops._LOCALIZATION, "submit", return_value=(True, "accepted")) as submit,
        ):
            result = map_ops.pose_estimate_impl(1, 2, .3, run_id=5)
        self.assertTrue(result["ok"])
        submit.assert_called_once_with((1.,2.,.3), 5)

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
            patch.object(map_ops.time, "sleep"),
        ):
            result = map_ops.pose_estimate_impl(1.0, 2.0, 0.3)
        self.assertTrue(result["ok"])
        self.assertIn(map_ops.INITIALPOSE_TOPIC, result["detail"])
        self.assertEqual(len(published), 1)

    def test_rtabmap_cannot_start_initial_localization(self):
        from mapping_rbnx import webui
        with (
            patch.object(map_ops, "active_algo", return_value="rtabmap"),
            patch.object(map_ops.lifecycle, "mark_reset") as reset,
            patch.object(map_ops._LOCALIZATION, "start") as start,
            patch.object(map_ops._LOCALIZATION, "snapshot") as snapshot,
        ):
            self.assertFalse(map_ops.relocalize_impl()["ok"])
            self.assertFalse(map_ops._begin_relocalization(None, None, "", "lab", None)[0])
            self.assertEqual(webui._localization_state(), {"state": "off"})
            webui._enforce_trust(False, "lab", 0.0)
            reset.assert_not_called()
            start.assert_not_called()
            snapshot.assert_not_called()

    def test_selected_pose_requires_a_live_scan_check(self):
        with patch.object(map_ops.localizers, "scan_fit", return_value=(-1.0, "no fresh scan")):
            ok, detail = map_ops._confirm_against_map(object(), object(), (1, 2, 0),
                                                       require_scan=True)
        self.assertFalse(ok)
        self.assertIn("no fresh scan", detail)


if __name__ == "__main__":
    unittest.main()
