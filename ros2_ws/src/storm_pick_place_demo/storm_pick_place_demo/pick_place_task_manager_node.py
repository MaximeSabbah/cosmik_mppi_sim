from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import rcl_interfaces.msg
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener

from .franka_gripper_client import FrankaGripperClient


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = float(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_msg_to_se3(tf) -> pin.SE3:
    R = _quat_to_matrix(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    t = np.array([tf.translation.x, tf.translation.y, tf.translation.z], dtype=np.float64)
    return pin.SE3(R, t)


def se3_to_pose_msg(M: pin.SE3):
    from geometry_msgs.msg import Pose

    xyzquat = pin.SE3ToXYZQUAT(M)
    msg = Pose()
    msg.position.x = float(xyzquat[0])
    msg.position.y = float(xyzquat[1])
    msg.position.z = float(xyzquat[2])
    msg.orientation.x = float(xyzquat[3])
    msg.orientation.y = float(xyzquat[4])
    msg.orientation.z = float(xyzquat[5])
    msg.orientation.w = float(xyzquat[6])
    return msg


class Phase(str, Enum):
    IDLE = "idle"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    CLOSE = "close_gripper"
    LIFT = "lift"
    PREPLACE = "preplace"
    PLACE = "place"
    OPEN = "open_gripper"
    RETREAT = "retreat"
    DONE = "done"


class TaskMode(str, Enum):
    PICK_PLACE = "pick_place"
    PLACE_PLACE = "place_place"


@dataclass
class TaskParams:
    goal_topic: str
    task_mode: TaskMode
    base_frame: str
    ee_frame: str
    pick_frame: str
    pick_xyzquat: np.ndarray
    place_frame: str
    place_xyzquat: np.ndarray
    pregrasp_offset: np.ndarray
    lift_offset: np.ndarray
    retreat_offset: np.ndarray
    pos_tolerance: float
    rot_tolerance_rad: float
    pregrasp_preplace_pos_tolerance: float
    pregrasp_preplace_rot_tolerance_rad: float
    grasp_place_pos_tolerance: float
    grasp_place_rot_tolerance_rad: float
    phase_timeout_sec: float
    target_timeout_sec: float
    loop_forever: bool
    max_cycles: int
    shutdown_on_done: bool
    auto_start: bool
    armed: bool
    disarm_bridge_during_gripper: bool
    gripper_open_width: float
    gripper_open_effort: float
    gripper_close_width: float
    gripper_close_effort: float
    wait_for_gripper_result: bool
    gripper_open_wait_sec: float
    gripper_close_wait_sec: float


class PickPlaceTaskManagerNode(Node):
    """Sequences a pick-and-place task by publishing goals and operating the gripper."""

    def __init__(self) -> None:
        super().__init__("pick_place_task_manager")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._declare_parameters()
        self._params = self._get_params()

        self._arm_id = self._derive_arm_id(self._params.base_frame, self._params.ee_frame)
        self.get_logger().info(f"Using arm_id '{self._arm_id}'.")

        self._gripper = FrankaGripperClient(self, arm_id=self._arm_id)

        bridge_node = str(self.get_parameter("bridge_node_name").value)
        self._bridge_set_params_client = self.create_client(
            SetParameters, f"/{bridge_node}/set_parameters"
        )

        self._goal_pub = self.create_publisher(
            PoseStamped,
            self._params.goal_topic,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        self._phase: Phase = Phase.IDLE
        self._phase_start_wall: Optional[float] = None
        self._timed_target: Optional[tuple[str, tuple[float, ...]]] = None
        self._target_start_monotonic: Optional[float] = None
        self._current_goal_M: Optional[pin.SE3] = None
        self._forward_pick_to_place: bool = True
        self._completed_cycles: int = 0
        self._gripper_goal_future: Optional[Any] = None
        self._gripper_result_future: Optional[Any] = None
        self._pose_a_frame: str = self._params.pick_frame
        self._pose_a_xyzquat: np.ndarray = self._params.pick_xyzquat.copy()
        self._pose_b_frame: str = self._params.place_frame
        self._pose_b_xyzquat: np.ndarray = self._params.place_xyzquat.copy()

        self._waypoints_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("waypoints_markers_topic").value),
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._waypoints_timer = self.create_timer(0.5, self._publish_waypoint_markers)

        self._loop_timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"PickPlace task manager ready. mode={self._params.task_mode.value} "
            f"armed={self._params.armed} auto_start={self._params.auto_start}"
        )

    def _target_now(self) -> float:
        if self.get_parameter("use_sim_time").value:
            return self.get_clock().now().nanoseconds * 1.0e-9
        return time.monotonic()

    def _phase_now(self) -> float:
        if self.get_parameter("use_sim_time").value:
            return self.get_clock().now().nanoseconds * 1.0e-9
        return time.time()

    def _read_bool_param(self, name: str) -> bool:
        v = self.get_parameter(name).value
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
        return bool(v)

    @staticmethod
    def _derive_arm_id(base_frame: str, ee_frame: str) -> str:
        for token in ("_hand", "_link"):
            if token in ee_frame:
                return ee_frame.split(token)[0]
        if "_link" in base_frame:
            return base_frame.split("_link")[0]
        raise RuntimeError(
            f"Unable to derive arm_id from base_frame='{base_frame}' ee_frame='{ee_frame}'."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "goal_topic",
            "/goal_pose",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="PoseStamped topic used as goal input for the high-level planner.",
            ),
        )
        self.declare_parameter(
            "task_mode",
            TaskMode.PICK_PLACE.value,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Task mode: pick_place or place_place.",
            ),
        )
        self.declare_parameter("base_frame", "fer_link0")
        self.declare_parameter("ee_frame", "fer_hand_tcp")

        self.declare_parameter(
            "auto_start",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, start the task automatically (pregrasp -> ...).",
            ),
        )
        self.declare_parameter(
            "armed",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Safety gate. When false, the node publishes goals but does not command the gripper."
            ),
        )

        self.declare_parameter("pick_frame", "fer_link0")
        self.declare_parameter(
            "pick_xyzquat",
            [0.60, -0.30, 0.20, 0.0, 1.0, 0.0, 0.0],
            ParameterDescriptor(description="Pick pose as [x,y,z,qx,qy,qz,qw]."),
        )
        self.declare_parameter("place_frame", "fer_link0")
        self.declare_parameter(
            "place_xyzquat",
            [0.40, 0.30, 0.20, 0.0, 1.0, 0.0, 0.0],
            ParameterDescriptor(description="Place pose as [x,y,z,qx,qy,qz,qw]."),
        )

        self.declare_parameter("pregrasp_offset", [0.0, 0.0, 0.10])
        self.declare_parameter("lift_offset", [0.0, 0.0, 0.15])
        self.declare_parameter("retreat_offset", [0.0, 0.0, 0.10])

        self.declare_parameter(
            "enable_waypoint_visualization",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Publish small labeled markers at every phase waypoint (pregrasp/grasp/lift/preplace/place/retreat), complementing object_manager_node's pick/place object cubes.",
            ),
        )
        self.declare_parameter(
            "waypoints_markers_topic",
            "/storm_pick_place/waypoints_markers",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )

        self.declare_parameter("pos_tolerance", 0.02)
        self.declare_parameter("rot_tolerance_rad", 0.6)
        self.declare_parameter(
            "pregrasp_preplace_pos_tolerance",
            -1.0,
            ParameterDescriptor(
                description="Position tolerance for PREGRASP/PREPLACE/LIFT/RETREAT phases. <=0 uses pos_tolerance."
            ),
        )
        self.declare_parameter(
            "pregrasp_preplace_rot_tolerance_rad",
            -1.0,
            ParameterDescriptor(
                description="Rotation tolerance for PREGRASP/PREPLACE/LIFT/RETREAT phases. <=0 uses rot_tolerance_rad."
            ),
        )
        self.declare_parameter(
            "grasp_place_pos_tolerance",
            -1.0,
            ParameterDescriptor(
                description="Position tolerance for GRASP/PLACE phases. <=0 uses pos_tolerance."
            ),
        )
        self.declare_parameter(
            "grasp_place_rot_tolerance_rad",
            -1.0,
            ParameterDescriptor(
                description="Rotation tolerance for GRASP/PLACE phases. <=0 uses rot_tolerance_rad."
            ),
        )
        self.declare_parameter("phase_timeout_sec", 60.0)
        self.declare_parameter(
            "target_timeout_sec",
            0.0,
            ParameterDescriptor(
                description=(
                    "Reaching-demo time limit per A/B target, across its motion phases. "
                    "On expiry, skip the remaining motion phases and pursue the other target. "
                    "Gripper actions finish before switching. <=0 disables this behavior."
                )
            ),
        )
        self.declare_parameter("loop_forever", True)
        self.declare_parameter(
            "max_cycles",
            0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Maximum completed cycles before transitioning to DONE. <=0 disables the cap.",
            ),
        )
        self.declare_parameter(
            "shutdown_on_done",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, shutdown this node process after entering DONE.",
            ),
        )

        self.declare_parameter(
            "bridge_node_name",
            "storm_mppi_planner",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Node name of the STORM bridge — used to flip armed=false (hold) during gripper phases.",
            ),
        )
        self.declare_parameter(
            "disarm_bridge_during_gripper",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description=(
                    "When true (default), sets the STORM bridge's armed=false during CLOSE/OPEN "
                    "gripper phases and armed=true again at LIFT/RETREAT, holding the arm still "
                    "while the gripper closes/opens. When false, the bridge is left armed "
                    "continuously through the whole pick/place cycle and the arm is not held "
                    "during gripper motion."
                ),
            ),
        )

        self.declare_parameter("gripper_open_width", 0.039)
        self.declare_parameter("gripper_open_effort", 10.0)
        self.declare_parameter("gripper_close_width", 0.01)
        self.declare_parameter("gripper_close_effort", 30.0)
        self.declare_parameter("wait_for_gripper_result", False)
        self.declare_parameter("gripper_open_wait_sec", 1.0)
        self.declare_parameter("gripper_close_wait_sec", 1.0)

    def _get_params(self) -> TaskParams:
        def b(name: str) -> bool:
            v = self.get_parameter(name).value
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
            return bool(v)

        def arr(name: str, n: int) -> np.ndarray:
            v = np.array(self.get_parameter(name).value, dtype=np.float64)
            if v.size != n:
                raise ValueError(f"{name} must have length {n}, got {v.size}")
            return v

        task_mode = self._parse_task_mode(self.get_parameter("task_mode").value)
        pos_tolerance = float(self.get_parameter("pos_tolerance").value)
        rot_tolerance_rad = float(self.get_parameter("rot_tolerance_rad").value)

        pregrasp_preplace_pos_tolerance = float(self.get_parameter("pregrasp_preplace_pos_tolerance").value)
        if pregrasp_preplace_pos_tolerance <= 0.0:
            pregrasp_preplace_pos_tolerance = pos_tolerance

        pregrasp_preplace_rot_tolerance_rad = float(
            self.get_parameter("pregrasp_preplace_rot_tolerance_rad").value
        )
        if pregrasp_preplace_rot_tolerance_rad <= 0.0:
            pregrasp_preplace_rot_tolerance_rad = rot_tolerance_rad

        grasp_place_pos_tolerance = float(self.get_parameter("grasp_place_pos_tolerance").value)
        if grasp_place_pos_tolerance <= 0.0:
            grasp_place_pos_tolerance = pos_tolerance

        grasp_place_rot_tolerance_rad = float(self.get_parameter("grasp_place_rot_tolerance_rad").value)
        if grasp_place_rot_tolerance_rad <= 0.0:
            grasp_place_rot_tolerance_rad = rot_tolerance_rad

        return TaskParams(
            goal_topic=str(self.get_parameter("goal_topic").value),
            task_mode=task_mode,
            base_frame=str(self.get_parameter("base_frame").value),
            ee_frame=str(self.get_parameter("ee_frame").value),
            pick_frame=str(self.get_parameter("pick_frame").value),
            pick_xyzquat=arr("pick_xyzquat", 7),
            place_frame=str(self.get_parameter("place_frame").value),
            place_xyzquat=arr("place_xyzquat", 7),
            pregrasp_offset=arr("pregrasp_offset", 3),
            lift_offset=arr("lift_offset", 3),
            retreat_offset=arr("retreat_offset", 3),
            pos_tolerance=pos_tolerance,
            rot_tolerance_rad=rot_tolerance_rad,
            pregrasp_preplace_pos_tolerance=pregrasp_preplace_pos_tolerance,
            pregrasp_preplace_rot_tolerance_rad=pregrasp_preplace_rot_tolerance_rad,
            grasp_place_pos_tolerance=grasp_place_pos_tolerance,
            grasp_place_rot_tolerance_rad=grasp_place_rot_tolerance_rad,
            phase_timeout_sec=float(self.get_parameter("phase_timeout_sec").value),
            target_timeout_sec=float(self.get_parameter("target_timeout_sec").value),
            loop_forever=b("loop_forever"),
            max_cycles=int(self.get_parameter("max_cycles").value),
            shutdown_on_done=b("shutdown_on_done"),
            auto_start=b("auto_start"),
            armed=b("armed"),
            disarm_bridge_during_gripper=b("disarm_bridge_during_gripper"),
            gripper_open_width=float(self.get_parameter("gripper_open_width").value),
            gripper_open_effort=float(self.get_parameter("gripper_open_effort").value),
            gripper_close_width=float(self.get_parameter("gripper_close_width").value),
            gripper_close_effort=float(self.get_parameter("gripper_close_effort").value),
            wait_for_gripper_result=b("wait_for_gripper_result"),
            gripper_open_wait_sec=float(self.get_parameter("gripper_open_wait_sec").value),
            gripper_close_wait_sec=float(self.get_parameter("gripper_close_wait_sec").value),
        )

    @staticmethod
    def _parse_task_mode(value: object) -> TaskMode:
        mode_str = str(value).strip().lower()
        try:
            return TaskMode(mode_str)
        except ValueError as exc:
            valid = ", ".join(m.value for m in TaskMode)
            raise ValueError(f"Invalid task_mode '{value}'. Valid values: {valid}") from exc

    def _initial_phase(self) -> Phase:
        if self._params.task_mode == TaskMode.PLACE_PLACE:
            return Phase.PREPLACE
        return Phase.PREGRASP

    def _target_label(self, frame: str, xyzquat: np.ndarray) -> str:
        if frame == self._pose_a_frame and np.allclose(xyzquat, self._pose_a_xyzquat, atol=1.0e-6):
            return "A"
        if frame == self._pose_b_frame and np.allclose(xyzquat, self._pose_b_xyzquat, atol=1.0e-6):
            return "B"
        return "?"

    def _pick_place_labels(self) -> tuple[str, str]:
        return (
            self._target_label(self._params.pick_frame, self._params.pick_xyzquat),
            self._target_label(self._params.place_frame, self._params.place_xyzquat),
        )

    def _phase_event_label(self, phase: Phase) -> str:
        if self._params.task_mode == TaskMode.PLACE_PLACE:
            target_frame, target_xyzquat = self._active_place_target()
            target_label = self._target_label(target_frame, target_xyzquat)
            if phase == Phase.PREPLACE:
                return f"preplace({target_label})"
            if phase == Phase.PLACE:
                return f"place({target_label})"
            if phase == Phase.RETREAT:
                return f"lift({target_label})"
            return phase.value

        pick_label, place_label = self._pick_place_labels()
        if phase == Phase.PREGRASP:
            return f"pregrasp({pick_label})"
        if phase == Phase.GRASP:
            return f"grasp({pick_label})"
        if phase == Phase.CLOSE:
            return "close"
        if phase == Phase.LIFT:
            return f"lift({pick_label})"
        if phase == Phase.PREPLACE:
            return f"preplace({place_label})"
        if phase == Phase.PLACE:
            return f"place({place_label})"
        if phase == Phase.OPEN:
            return "open"
        if phase == Phase.RETREAT:
            return f"lift({place_label})"
        return phase.value

    def _phase_tolerances(self, phase: Phase) -> tuple[float, float]:
        if phase in (Phase.GRASP, Phase.PLACE):
            return (
                self._params.grasp_place_pos_tolerance,
                self._params.grasp_place_rot_tolerance_rad,
            )
        if phase in (Phase.PREGRASP, Phase.PREPLACE, Phase.LIFT, Phase.RETREAT):
            return (
                self._params.pregrasp_preplace_pos_tolerance,
                self._params.pregrasp_preplace_rot_tolerance_rad,
            )
        return (self._params.pos_tolerance, self._params.rot_tolerance_rad)

    def _current_target(self) -> tuple[str, np.ndarray]:
        if self._params.task_mode == TaskMode.PLACE_PLACE:
            return self._active_place_target()
        if self._phase in (Phase.PREGRASP, Phase.GRASP, Phase.CLOSE, Phase.LIFT):
            return self._params.pick_frame, self._params.pick_xyzquat
        return self._params.place_frame, self._params.place_xyzquat

    def _target_timed_out(self) -> bool:
        # Track the physical target, not phase names or mutable pick/place roles.
        # In particular, RETREAT(B) -> PREGRASP(B) must retain B's deadline.
        frame, xyzquat = self._current_target()
        target = (frame, tuple(float(v) for v in xyzquat))
        now = self._target_now()
        if target != self._timed_target or self._target_start_monotonic is None:
            self._timed_target = target
            self._target_start_monotonic = now
        timeout = self._params.target_timeout_sec
        if now < self._target_start_monotonic:
            self._target_start_monotonic = now
        elapsed = now - self._target_start_monotonic
        if timeout <= 0.0 or elapsed < timeout:
            return False
        self.get_logger().info(
            f"[task-event] target-timeout({self._target_label(frame, xyzquat)}, "
            f"elapsed={elapsed:.2f}s); advancing to the other target."
        )
        return True

    def _advance_after_target_timeout(self) -> None:
        if self._params.task_mode == TaskMode.PLACE_PLACE:
            self._advance_from_place_place(Phase.RETREAT)
        elif self._phase in (Phase.PREGRASP, Phase.GRASP, Phase.LIFT):
            self._enter_phase(Phase.PREPLACE)
        else:
            self._completed_cycles += 1
            if self._cycle_cap_reached() or not self._params.loop_forever:
                self._enter_phase(Phase.DONE)
            else:
                # The normal cycle swaps pick/place and visits the same pose
                # again. A target timeout must go straight to the opposite pose.
                self._enter_phase(Phase.PREGRASP)

    def _tick(self) -> None:
        # Refresh params that users may toggle at runtime.
        self._params.armed = self._read_bool_param("armed")
        self._params.auto_start = self._read_bool_param("auto_start")
        self._params.loop_forever = self._read_bool_param("loop_forever")
        self._params.shutdown_on_done = self._read_bool_param("shutdown_on_done")

        if self._phase == Phase.IDLE:
            if self._params.auto_start:
                self._enter_phase(self._initial_phase())
            return

        if self._phase in (Phase.DONE,):
            return

        if self._phase_start_wall is None:
            self._phase_start_wall = self._phase_now()

        timeout_sec = self._params.phase_timeout_sec
        if timeout_sec > 0.0 and (self._phase_now() - self._phase_start_wall) > timeout_sec:
            self.get_logger().error(f"Phase timeout in {self._phase}. Aborting.")
            self._enter_phase(Phase.DONE)
            return

        if self._phase in (Phase.CLOSE, Phase.OPEN):
            if self._params.armed and self._gripper_goal_future is None:
                if self._phase == Phase.OPEN:
                    self._open_gripper(track_result=True, reason="retry_open")
                else:
                    self._close_gripper(track_result=True, reason="retry_close")

            if self._is_gripper_phase_complete(self._phase, require_goal=self._params.armed):
                next_phase = Phase.LIFT if self._phase == Phase.CLOSE else Phase.RETREAT
                self._enter_phase(next_phase)
            return

        if self._phase == Phase.PREGRASP and self._params.armed:
            # Always enforce: OPEN must happen before any pregrasp motion.
            if self._gripper_goal_future is None:
                self._open_gripper(track_result=True, reason="retry_open_before_pregrasp")
                return
            if not self._is_gripper_phase_complete(Phase.OPEN, require_goal=True):
                return

        # Publish current phase goal.
        goal_M = self._compute_phase_goal(self._phase)
        if goal_M is None:
            return
        # Start the deadline when a valid goal is available. Gripper phases above
        # finish normally; their elapsed time still counts toward this target.
        if self._target_timed_out():
            self._advance_after_target_timeout()
            return
        self._current_goal_M = goal_M
        self._publish_goal(goal_M)

        # Check convergence.
        ee_M = self._lookup_ee_in_base()
        if ee_M is None:
            return

        pos_err = np.linalg.norm(ee_M.translation - goal_M.translation)
        rot_vec = pin.log3(goal_M.rotation.T @ ee_M.rotation)
        rot_err = float(np.linalg.norm(rot_vec))
        pos_tol, rot_tol = self._phase_tolerances(self._phase)

        if pos_err < pos_tol and rot_err < rot_tol:
            self._advance_from(self._phase)

    def _enter_phase(self, phase: Phase) -> None:
        self._phase = phase
        self._phase_start_wall = self._phase_now()
        if phase in (Phase.IDLE, Phase.DONE):
            self._timed_target = None
            self._target_start_monotonic = None
        if phase not in (Phase.CLOSE, Phase.OPEN):
            self._clear_gripper_action_state()

        if phase == Phase.PREGRASP:
            self._open_gripper(track_result=True, reason="before_pregrasp")
            if self._params.armed:
                self.get_logger().info("[task-event] open")
        elif phase == Phase.CLOSE:
            if self._params.disarm_bridge_during_gripper:
                self._set_bridge_armed(False)
            self._close_gripper(track_result=True, reason="pick")
        elif phase == Phase.OPEN:
            if self._params.disarm_bridge_during_gripper:
                self._set_bridge_armed(False)
            self._open_gripper(track_result=True, reason="place")
        elif phase in (Phase.LIFT, Phase.RETREAT):
            if self._params.disarm_bridge_during_gripper and self._params.armed:
                self._set_bridge_armed(True)
        self.get_logger().info(f"[task-event] {self._phase_event_label(phase)}")
        if phase == Phase.DONE and self._params.shutdown_on_done:
            self.get_logger().info("Task reached DONE; shutting down task manager node.")
            rclpy.shutdown()

    def _cycle_cap_reached(self) -> bool:
        return self._params.max_cycles > 0 and self._completed_cycles >= self._params.max_cycles

    def _clear_gripper_action_state(self) -> None:
        self._gripper_goal_future = None
        self._gripper_result_future = None

    def _set_bridge_armed(self, armed: bool) -> None:
        if not self._bridge_set_params_client.service_is_ready():
            self.get_logger().warn(
                f"Bridge set_parameters service not ready; cannot set armed={armed}.",
                throttle_duration_sec=5.0,
            )
            return
        pv = ParameterValue()
        pv.type = Parameter.Type.BOOL.value
        pv.bool_value = armed
        req = SetParameters.Request()
        p = rcl_interfaces.msg.Parameter()
        p.name = "armed"
        p.value = pv
        req.parameters = [p]
        future = self._bridge_set_params_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().debug(f"Bridge armed={armed} set: {f.result()}")
        )

    def _open_gripper(self, *, track_result: bool, reason: str) -> None:
        if not self._params.armed:
            return
        goal_future = self._gripper.command(
            position=self._params.gripper_open_width,
            max_effort=self._params.gripper_open_effort,
        )
        if track_result:
            self._gripper_goal_future = goal_future
            self._gripper_result_future = None

    def _close_gripper(self, *, track_result: bool, reason: str) -> None:
        if not self._params.armed:
            return
        res = self._gripper.grasp(
            width=self._params.gripper_close_width,
            force=self._params.gripper_close_effort,
            timeout_sec=0.2,
        )
        if res is None:
            fallback = self._gripper.command(
                position=self._params.gripper_close_width,
                max_effort=self._params.gripper_close_effort,
            )
            if track_result:
                self._gripper_goal_future = fallback
                self._gripper_result_future = None
            return
        if track_result:
            self._gripper_goal_future = res
            self._gripper_result_future = None

    def _gripper_phase_wait_sec(self, phase: Phase) -> float:
        if phase == Phase.CLOSE:
            return max(0.0, float(self._params.gripper_close_wait_sec))
        if phase == Phase.OPEN:
            return max(0.0, float(self._params.gripper_open_wait_sec))
        return 0.0

    def _is_gripper_phase_complete(self, phase: Phase, *, require_goal: bool = False) -> bool:
        if self._phase_start_wall is None:
            return False
        elapsed = self._phase_now() - self._phase_start_wall
        wait_sec = self._gripper_phase_wait_sec(phase)
        fallback_wait = wait_sec if wait_sec > 0.0 else 1.0

        if require_goal and self._gripper_goal_future is None:
            return False

        if not self._params.wait_for_gripper_result:
            return elapsed >= fallback_wait

        if self._gripper_goal_future is None:
            return elapsed >= fallback_wait

        if not self._gripper_goal_future.done():
            return elapsed >= fallback_wait

        if self._gripper_result_future is None:
            try:
                goal_handle = self._gripper_goal_future.result()
            except Exception as exc:
                self.get_logger().warn(f"Gripper goal future failed: {exc}")
                self._clear_gripper_action_state()
                return False
            if goal_handle is None:
                self._clear_gripper_action_state()
                return False
            if not goal_handle.accepted:
                self.get_logger().warn("Gripper goal rejected; retrying.")
                self._clear_gripper_action_state()
                return False
            self._gripper_result_future = goal_handle.get_result_async()

        if self._gripper_result_future.done():
            return True
        return elapsed >= fallback_wait

    def _advance_from(self, phase: Phase) -> None:
        if self._params.task_mode == TaskMode.PLACE_PLACE:
            self._advance_from_place_place(phase)
            return

        self._advance_from_pick_place(phase)

    def _advance_from_pick_place(self, phase: Phase) -> None:
        if phase == Phase.PREGRASP:
            self._enter_phase(Phase.GRASP)
        elif phase == Phase.GRASP:
            self._enter_phase(Phase.CLOSE)
        elif phase == Phase.LIFT:
            self._enter_phase(Phase.PREPLACE)
        elif phase == Phase.PREPLACE:
            self._enter_phase(Phase.PLACE)
        elif phase == Phase.PLACE:
            self._enter_phase(Phase.OPEN)
        elif phase == Phase.RETREAT:
            self._completed_cycles += 1
            if self._cycle_cap_reached():
                self._enter_phase(Phase.DONE)
            elif self._params.loop_forever:
                self._swap_pick_place_targets()
                pick_label, place_label = self._pick_place_labels()
                self.get_logger().info(
                    f"[task-event] swap(cycle={self._completed_cycles}, pick={pick_label}, place={place_label})"
                )
                self._enter_phase(Phase.PREGRASP)
            else:
                self._enter_phase(Phase.DONE)

    def _advance_from_place_place(self, phase: Phase) -> None:
        if phase == Phase.PREPLACE:
            self._enter_phase(Phase.PLACE)
        elif phase == Phase.PLACE:
            self._enter_phase(Phase.RETREAT)
        elif phase == Phase.RETREAT:
            self._completed_cycles += 1
            if self._cycle_cap_reached():
                self._enter_phase(Phase.DONE)
            elif self._params.loop_forever:
                self._forward_pick_to_place = not self._forward_pick_to_place
                current_target = "A" if self._forward_pick_to_place else "B"
                self.get_logger().info(
                    f"Completed cycle {self._completed_cycles}. Switching place target to {current_target}."
                )
                self._enter_phase(Phase.PREPLACE)
            else:
                self._enter_phase(Phase.DONE)

    def _compute_phase_goal(self, phase: Phase) -> Optional[pin.SE3]:
        if self._params.task_mode == TaskMode.PLACE_PLACE:
            return self._compute_place_place_goal(phase)

        pick_frame, pick_xyzquat, place_frame, place_xyzquat = self._active_pick_place_targets()

        if phase == Phase.PREGRASP:
            return self._lookup_pose_in_base(pick_frame, pick_xyzquat, self._params.pregrasp_offset)
        if phase == Phase.GRASP:
            return self._lookup_pose_in_base(pick_frame, pick_xyzquat, np.zeros(3))
        if phase == Phase.LIFT:
            return self._lookup_pose_in_base(pick_frame, pick_xyzquat, self._params.lift_offset)
        if phase == Phase.PREPLACE:
            return self._lookup_pose_in_base(place_frame, place_xyzquat, self._params.pregrasp_offset)
        if phase == Phase.PLACE:
            return self._lookup_pose_in_base(place_frame, place_xyzquat, np.zeros(3))
        if phase == Phase.RETREAT:
            # In pick_place mode this is the "lift place" step.
            return self._lookup_pose_in_base(place_frame, place_xyzquat, self._params.retreat_offset)
        return None

    def _compute_place_place_goal(self, phase: Phase) -> Optional[pin.SE3]:
        target_frame, target_xyzquat = self._active_place_target()
        if phase == Phase.PREPLACE:
            return self._lookup_pose_in_base(target_frame, target_xyzquat, self._params.pregrasp_offset)
        if phase == Phase.PLACE:
            return self._lookup_pose_in_base(target_frame, target_xyzquat, np.zeros(3))
        if phase == Phase.RETREAT:
            return self._lookup_pose_in_base(target_frame, target_xyzquat, self._params.retreat_offset)
        return None

    def _active_pick_place_targets(self) -> tuple[str, np.ndarray, str, np.ndarray]:
        return (
            self._params.pick_frame,
            self._params.pick_xyzquat,
            self._params.place_frame,
            self._params.place_xyzquat,
        )

    def _active_place_target(self) -> tuple[str, np.ndarray]:
        if self._forward_pick_to_place:
            return (self._params.pick_frame, self._params.pick_xyzquat)
        return (self._params.place_frame, self._params.place_xyzquat)

    _WAYPOINT_STYLE: dict[Phase, tuple[float, float, float]] = {
        Phase.PREGRASP: (1.0, 1.0, 0.0),
        Phase.GRASP: (1.0, 0.5, 0.0),
        Phase.LIFT: (0.0, 1.0, 1.0),
        Phase.PREPLACE: (0.6, 0.0, 1.0),
        Phase.PLACE: (0.0, 0.8, 0.2),
        Phase.RETREAT: (1.0, 0.0, 1.0),
    }

    def _publish_waypoint_markers(self) -> None:
        """Small labeled spheres at every phase waypoint (pregrasp/grasp/lift/preplace/place/
        retreat), reusing _compute_phase_goal exactly as the state machine does so this can
        never drift from the actual targets. Complements object_manager_node's pick/place
        cubes, which only show the two end objects, not the offsets in between. Runs on its own
        slow timer (2 Hz) -- these only change when targets swap, no need to match the 20Hz
        state-machine tick.
        """
        if not bool(self.get_parameter("enable_waypoint_visualization").value):
            return

        if self._params.task_mode == TaskMode.PICK_PLACE:
            phases = [Phase.PREGRASP, Phase.GRASP, Phase.LIFT, Phase.PREPLACE, Phase.PLACE, Phase.RETREAT]
        else:
            phases = [Phase.PREPLACE, Phase.PLACE, Phase.RETREAT]

        now = self.get_clock().now().to_msg()
        markers: list[Marker] = []
        for i, phase in enumerate(phases):
            try:
                goal_M = self._compute_phase_goal(phase)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"Waypoint marker skipped for {phase.value} (non-fatal): {exc}", throttle_duration_sec=5.0
                )
                continue
            if goal_M is None:
                continue
            pose_msg = se3_to_pose_msg(goal_M)
            color = self._WAYPOINT_STYLE.get(phase, (0.8, 0.8, 0.8))

            sphere = Marker()
            sphere.header.stamp = now
            sphere.header.frame_id = self._params.base_frame
            sphere.ns = "pp_waypoints"
            sphere.id = 2 * i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = pose_msg
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.025
            sphere.color.r, sphere.color.g, sphere.color.b = color
            sphere.color.a = 0.9
            sphere.lifetime.sec = 1
            markers.append(sphere)

            label = Marker()
            label.header.stamp = now
            label.header.frame_id = self._params.base_frame
            label.ns = "pp_waypoints_labels"
            label.id = 2 * i + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose = pose_msg
            label.pose.position.z = pose_msg.position.z + 0.04
            label.scale.z = 0.025
            label.color.r, label.color.g, label.color.b = color
            label.color.a = 1.0
            label.text = phase.value
            label.lifetime.sec = 1
            markers.append(label)

        if markers:
            self._waypoints_pub.publish(MarkerArray(markers=markers))

    def _swap_pick_place_targets(self) -> None:
        pick_frame = self._params.pick_frame
        pick_xyzquat = self._params.pick_xyzquat.copy()
        self._params.pick_frame = self._params.place_frame
        self._params.pick_xyzquat = self._params.place_xyzquat.copy()
        self._params.place_frame = pick_frame
        self._params.place_xyzquat = pick_xyzquat

    def _lookup_pose_in_base(self, frame: str, xyzquat: np.ndarray, offset: np.ndarray) -> Optional[pin.SE3]:
        # Pose in the requested frame.
        M_local = pin.XYZQUATToSE3(xyzquat)
        M_local.translation = M_local.translation + offset

        if frame == self._params.base_frame:
            return M_local

        try:
            tf = self._tf_buffer.lookup_transform(self._params.base_frame, frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed {frame}->{self._params.base_frame}: {exc}")
            return None

        base_T_frame = transform_msg_to_se3(tf.transform)
        return base_T_frame * M_local

    def _lookup_ee_in_base(self) -> Optional[pin.SE3]:
        try:
            tf = self._tf_buffer.lookup_transform(self._params.base_frame, self._params.ee_frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed for ee: {exc}")
            return None
        return transform_msg_to_se3(tf.transform)

    def _publish_goal(self, goal_M: pin.SE3) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._params.base_frame
        msg.pose = se3_to_pose_msg(goal_M)
        self._goal_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceTaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
