"""Exercise the real task transitions with a fake clock and no ROS processes."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from storm_pick_place_demo import pick_place_task_manager_node as task


class TargetTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.wall = 0.0
        self.clock_patch = patch.object(task.time, "monotonic", lambda: self.now)
        self.wall_patch = patch.object(task.time, "time", lambda: self.wall)
        self.clock_patch.start()
        self.wall_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.addCleanup(self.wall_patch.stop)

        # Bypass Node.__init__: this test creates no publishers, clients or DDS
        # participants. Parameter parsing and task methods are the production code.
        node = task.PickPlaceTaskManagerNode.__new__(task.PickPlaceTaskManagerNode)
        values = {}
        node.declare_parameter = lambda name, default, *a, **kw: values.setdefault(name, default)
        node._declare_parameters()
        cfg = Path(__file__).resolve().parents[1] / "config/pick_place_task_manager_params.yaml"
        values.update(yaml.safe_load(cfg.read_text())["pick_place_task_manager"]["ros__parameters"])
        values.update(armed=False, auto_start=True, use_sim_time=False)
        node.get_parameter = lambda name: SimpleNamespace(value=values[name])
        node._params = node._get_params()
        node._phase = task.Phase.IDLE
        node._phase_start_wall = None
        node._timed_target = None
        node._target_start_monotonic = None
        node._current_goal_M = None
        node._forward_pick_to_place = True
        node._completed_cycles = 0
        node._gripper_goal_future = None
        node._gripper_result_future = None
        node._pose_a_frame = node._params.pick_frame
        node._pose_a_xyzquat = node._params.pick_xyzquat.copy()
        node._pose_b_frame = node._params.place_frame
        node._pose_b_xyzquat = node._params.place_xyzquat.copy()
        self.logger = Mock()
        node.get_logger = lambda: self.logger
        node._open_gripper = Mock()
        node._close_gripper = Mock()
        node._set_bridge_armed = Mock()
        node._lookup_ee_in_base = lambda: None
        self.goals = []
        node._publish_goal = lambda pose: self.goals.append(pose.translation.copy())
        self.node = node

    def tick(self, elapsed):
        self.now = elapsed
        self.wall = elapsed
        self.node._tick()

    def start(self):
        self.tick(0.0)  # IDLE -> first motion phase.
        self.tick(0.0)  # Publish first goal and start its deadline.

    def test_stalled_targets_alternate_at_ten_seconds(self):
        self.start()
        self.tick(9.99)
        self.assertEqual(self.goals[-1][1], -0.5)
        self.tick(10.0)
        self.assertEqual(self.node._phase, task.Phase.PREPLACE)
        self.tick(10.05)
        self.assertEqual(self.goals[-1][1], 0.5)
        self.tick(20.0)
        self.assertEqual(self.node._phase, task.Phase.PREPLACE)
        self.tick(20.1)
        self.assertEqual(self.node._phase, task.Phase.PREGRASP)
        self.tick(20.15)
        self.assertEqual(self.goals[-1][1], -0.5)

    def test_phase_changes_do_not_restart_target_deadline(self):
        self.start()
        self.tick(8.0)
        self.node._enter_phase(task.Phase.GRASP)
        self.tick(9.0)
        self.node._enter_phase(task.Phase.LIFT)
        self.tick(10.0)
        self.assertEqual(self.node._phase, task.Phase.PREPLACE)

    def test_role_swap_at_same_pose_keeps_deadline(self):
        self.node._enter_phase(task.Phase.PREPLACE)
        self.tick(0.0)
        self.tick(8.0)
        self.node._enter_phase(task.Phase.RETREAT)
        self.node._advance_from(task.Phase.RETREAT)
        self.assertEqual(self.node._phase, task.Phase.PREGRASP)
        self.tick(9.0)
        self.assertEqual(self.goals[-1][1], 0.5)
        self.tick(10.0)
        self.tick(10.05)
        self.assertEqual(self.goals[-1][1], -0.5)

    def test_spatial_arrival_advances_before_deadline(self):
        self.node._lookup_ee_in_base = lambda: self.node._compute_phase_goal(self.node._phase)
        self.start()
        self.tick(0.05)  # GRASP -> CLOSE
        self.tick(1.1)   # CLOSE -> LIFT
        self.tick(1.15)  # LIFT -> PREPLACE
        self.tick(1.2)   # Publish opposite target.
        self.assertEqual(self.goals[-1][1], 0.5)
        self.assertFalse(any("target-timeout" in str(c) for c in self.logger.info.call_args_list))

    def test_zero_disables_timed_alternation(self):
        self.node._params.target_timeout_sec = 0.0
        self.start()
        self.tick(100.0)
        self.assertEqual(self.node._phase, task.Phase.PREGRASP)
        self.assertEqual(self.goals[-1][1], -0.5)

    def test_gripper_phase_finishes_before_switching(self):
        self.start()
        self.tick(9.8)
        self.node._enter_phase(task.Phase.CLOSE)
        self.tick(10.0)
        self.assertEqual(self.node._phase, task.Phase.CLOSE)
        self.tick(10.9)
        self.assertEqual(self.node._phase, task.Phase.LIFT)
        self.tick(10.95)
        self.assertEqual(self.node._phase, task.Phase.PREPLACE)

    def test_wall_clock_adjustment_does_not_expire_target(self):
        self.start()
        self.now = 1.0
        self.wall = 10000.0
        self.node._tick()
        self.assertEqual(self.node._phase, task.Phase.PREGRASP)

    def test_cycle_cap_still_stops_after_place_timeout(self):
        self.node._params.max_cycles = 1
        self.node._enter_phase(task.Phase.PREPLACE)
        self.tick(0.0)
        self.tick(10.0)
        self.assertEqual(self.node._phase, task.Phase.DONE)

    def test_place_place_mode_also_switches(self):
        self.node._params.task_mode = task.TaskMode.PLACE_PLACE
        self.start()
        self.tick(10.0)
        self.tick(10.05)
        self.assertEqual(self.goals[-1][1], 0.5)

    def test_simulation_pause_does_not_consume_timeout(self):
        old_get = self.node.get_parameter
        self.node.get_parameter = lambda name: SimpleNamespace(value=True) if name == 'use_sim_time' else old_get(name)
        self.sim_time = 0.0
        self.node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.sim_time * 1e9)))
        self.start()
        self.tick(1000.0)  # wall time passes while Gazebo remains paused.
        self.assertEqual(self.node._phase, task.Phase.PREGRASP)
        self.sim_time = 10.01
        self.tick(1001.0)
        self.assertEqual(self.node._phase, task.Phase.PREPLACE)

    def test_backwards_simulation_time_restarts_target_deadline(self):
        old_get = self.node.get_parameter
        self.node.get_parameter = lambda name: SimpleNamespace(value=True) if name == 'use_sim_time' else old_get(name)
        self.sim_time = 100.0
        self.node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.sim_time * 1e9)))
        self.start()
        self.sim_time = 0.0
        self.tick(1.0)
        self.sim_time = 10.01
        self.tick(2.0)
        self.assertEqual(self.node._phase, task.Phase.PREPLACE)

    def test_existing_phase_abort_is_preserved(self):
        self.node._params.phase_timeout_sec = 5.0
        self.start()
        self.tick(5.1)
        self.assertEqual(self.node._phase, task.Phase.DONE)


if __name__ == "__main__":
    unittest.main()
