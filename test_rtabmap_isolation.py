# SPDX-License-Identifier: MulanPSL-2.0
"""RTAB-Map operations must work without touching initial-localization state."""
import unittest
from unittest.mock import Mock, patch

from mapping_rbnx import map_ops, webui
import test_map_load as load_cases
import test_map_session_state as session_tests
import test_pose_candidates as candidate_tests
import test_webui_api as api_tests


class _RtabmapOnly:
    def setUp(self):
        super().setUp()
        # A configured fallback must not divert RTAB-Map into initial localization.
        guards = [
            patch.object(map_ops, "active_algo", return_value="rtabmap"),
            patch.object(map_ops.localizers, "enabled", return_value=True),
        ]
        controller = Mock()
        for method in ("start", "submit", "snapshot", "cancel", "filter_running"):
            getattr(controller, method).side_effect = AssertionError(
                "RTAB-Map touched initial-localization controller: " + method)
        guards.append(patch.object(map_ops, "_LOCALIZATION", controller))
        for method in ("start", "stop", "match_statically", "wait_for_convergence"):
            guards.append(patch.object(map_ops.localizers, method, side_effect=AssertionError(
                "RTAB-Map touched initial-localization ROS adapter: " + method)))
        for guard in guards:
            guard.start()
            self.addCleanup(guard.stop)


class RtabmapLoadIsolationTest(_RtabmapOnly, load_cases.LoadMapTransactionTest):
    """Run existing load transaction checks with initial localization forbidden."""


class RtabmapDatabaseIsolationTest(_RtabmapOnly, session_tests.ActiveDatabaseRecordTest):
    """Runtime DB tracking and save must remain independent of the controller."""


class RtabmapResetIsolationTest(_RtabmapOnly, session_tests.ResetEpochTest):
    """Both reset success and partial failure retain their lifecycle semantics."""


class RtabmapEntryIsolationTest(_RtabmapOnly, unittest.TestCase):
    test_initialpose = candidate_tests.PoseCandidateTests.test_rtabmap_pose_estimate_keeps_the_initialpose_path

    def test_candidates_keep_the_original_unsupported_response(self):
        with patch.object(map_ops.lifecycle, "current", return_value={
            "map_id": "lab", "generation": 7, "mode": "localization",
        }):
            self.assertEqual(map_ops.get_pose_candidates_impl(), {
                "ok": False,
                "detail": "pose candidates are supported only by the slam_toolbox engine",
                "map_id": "lab", "generation": 7,
                "needs_selection": False, "candidates_json": "[]",
            })

    def test_relocalize_refuses_before_touching_ros_or_lifecycle(self):
        with (
            patch.object(map_ops, "_get_node") as node,
            patch.object(map_ops.lifecycle, "mark_reset") as reset,
        ):
            self.assertFalse(map_ops.relocalize_impl()["ok"])
            self.assertFalse(map_ops._begin_relocalization(None, None, "", "lab", None)[0])
            node.assert_not_called()
            reset.assert_not_called()

    def test_mode_switches_keep_the_engine_service_path(self):
        for mode in ("mapping", "localization"):
            with (
                self.subTest(mode=mode),
                patch.object(map_ops, "_get_node", return_value=object()),
                patch.object(map_ops.lifecycle, "current", return_value={
                    "map_id": "lab", "mode": "localization"}),
                patch.object(map_ops, "_call_service", return_value=(True, "ok")) as switch,
                patch.object(map_ops.lifecycle, "set_mode") as publish,
            ):
                self.assertTrue(map_ops.switch_mode_impl(mode)["ok"])
                self.assertEqual(switch.call_args.args[2], f"{map_ops.RTABMAP_NS}/set_mode_{mode}")
                publish.assert_called_once_with(mode)

    def test_ui_does_not_read_or_restart_initial_localization(self):
        self.assertEqual(webui._localization_state(), {"state": "off"})
        with patch.object(map_ops, "relocalize_impl") as restart:
            webui._enforce_trust(False, "lab", 0.0)
            restart.assert_not_called()


class RtabmapHttpIsolationTest(_RtabmapOnly, api_tests.WebUiApiTest):
    def test_pose_metadata_does_not_change_rtabmap_initialpose_forwarding(self):
        with (
            patch.object(map_ops, "pose_estimate_impl", return_value={"ok": True}) as estimate,
            patch.object(map_ops.lifecycle, "current") as current,
        ):
            out = self.call("/api/pose_estimate", {
                "x": 1, "y": 2, "theta": .3,
                "run_id": 99, "map_id": "outdated", "generation": -1,
            })
            self.assertTrue(out["ok"])
            estimate.assert_called_once_with(1.0, 2.0, .3)
            current.assert_not_called()


if __name__ == "__main__":
    unittest.main()
