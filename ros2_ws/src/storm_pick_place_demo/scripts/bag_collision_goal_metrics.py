#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


@dataclass
class SphereObs:
    name: str
    center: np.ndarray
    radius: float
    source: str


@dataclass
class CubeObs:
    name: str
    center: np.ndarray
    rotation: np.ndarray
    dims: np.ndarray
    source: str


@dataclass
class CylinderObs:
    name: str
    center: np.ndarray
    rotation: np.ndarray
    radius: float
    height: float
    source: str


@dataclass
class RobotSphereSpec:
    frame_id: int
    center_local: np.ndarray
    radius: float
    link_name: str


@dataclass
class RawMarkerObstacle:
    name: str
    source_frame: str
    marker_type: int
    pose_xyzquat: np.ndarray
    scale_xyz: np.ndarray


@dataclass
class RawPoseSphere:
    name: str
    source_frame: str
    center: np.ndarray
    radius: float


class OfflineTFBuffer:
    def __init__(self) -> None:
        self._static: dict[str, tuple[str, pin.SE3]] = {}
        self._dynamic: dict[str, tuple[str, pin.SE3, int]] = {}

    def update_tf_message(self, tf_msg, *, is_static: bool, bag_time_ns: int) -> None:
        for tf in tf_msg.transforms:
            parent = str(tf.header.frame_id)
            child = str(tf.child_frame_id)
            xyzquat = np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ],
                dtype=np.float64,
            )
            parent_M_child = pin.XYZQUATToSE3(xyzquat)
            if is_static:
                self._static[child] = (parent, parent_M_child)
            else:
                self._dynamic[child] = (parent, parent_M_child, int(bag_time_ns))

    def _parent_of(self, child: str) -> Optional[tuple[str, pin.SE3]]:
        if child in self._dynamic:
            parent, parent_M_child, _ = self._dynamic[child]
            return parent, parent_M_child
        if child in self._static:
            return self._static[child]
        return None

    def _root_M_frame(self, frame: str) -> Optional[tuple[str, pin.SE3]]:
        if not frame:
            return None
        current = frame
        visited = set()
        root_M_frame = pin.SE3.Identity()
        while True:
            if current in visited:
                return None
            visited.add(current)
            parent_info = self._parent_of(current)
            if parent_info is None:
                return current, root_M_frame
            parent, parent_M_current = parent_info
            root_M_frame = parent_M_current * root_M_frame
            current = parent

    def lookup_transform(self, target_frame: str, source_frame: str) -> Optional[pin.SE3]:
        if target_frame == source_frame:
            return pin.SE3.Identity()
        target_chain = self._root_M_frame(target_frame)
        source_chain = self._root_M_frame(source_frame)
        if target_chain is None or source_chain is None:
            return None
        target_root, root_M_target = target_chain
        source_root, root_M_source = source_chain
        if target_root != source_root:
            return None
        return root_M_target.inverse() * root_M_source


def _pose_to_xyzquat(pose_msg) -> np.ndarray:
    return np.array(
        [
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ],
        dtype=np.float64,
    )


def _robot_link_name_from_storm_link(link_name: str) -> str:
    if link_name.startswith("panda_link"):
        suffix = link_name.replace("panda_link", "")
        return f"fer_link{suffix}"
    if link_name == "panda_hand":
        return "fer_hand"
    return link_name


def _safe_sqrt(value: float) -> float:
    return math.sqrt(max(value, 0.0))


def _sdf_box(point_obj: np.ndarray, dims: np.ndarray) -> float:
    q = np.abs(point_obj) - 0.5 * dims
    outside = _safe_sqrt(float(np.sum(np.square(np.maximum(q, 0.0)))))
    inside = float(min(np.max(q), 0.0))
    return outside + inside


def _sdf_cylinder(point_obj: np.ndarray, radius: float, height: float) -> float:
    radial = math.hypot(float(point_obj[0]), float(point_obj[1])) - float(radius)
    axial = abs(float(point_obj[2])) - 0.5 * float(height)
    outside = _safe_sqrt(max(radial, 0.0) ** 2 + max(axial, 0.0) ** 2)
    inside = min(max(radial, axial), 0.0)
    return outside + inside


def _quat_rot_err_rad(goal_R: np.ndarray, current_R: np.ndarray) -> float:
    return float(np.linalg.norm(pin.log3(goal_R.T @ current_R)))


def _default_static_world() -> dict:
    return {
        "world_model": {
            "coll_objs": {
                "sphere": {},
                "cube": {
                    "brick1": {
                        "frame": "fer_link0",
                        "dims": [0.11, 0.22, 0.20],
                        "pose": [0.32, 0.0, 0.07, 0.0, 0.0, 0.0, 1.0],
                    },
                    "brick2": {
                        "frame": "fer_link0",
                        "dims": [0.11, 0.22, 0.20],
                        "pose": [0.63, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0],
                    },
                    "box1": {
                        "frame": "fer_link0",
                        "dims": [0.20, 0.35, 0.20],
                        "pose": [0.79, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0],
                    },
                    "box2": {
                        "frame": "fer_link0",
                        "dims": [0.20, 0.35, 0.20],
                        "pose": [0.47, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        }
    }


def _save_plots(
    rows: list[dict[str, object]],
    output_csv: Path,
    *,
    plot_mode_comparison: bool = False,
    plot_closest_hist: bool = False,
    include_static_goal_refs: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not found. Skipping plot generation.")
        return

    t = np.array([float(row["t_rel_s"]) for row in rows], dtype=np.float64)
    d_min = np.array([float(row["min_collision_distance_m"]) for row in rows], dtype=np.float64)
    has_two_mode_cols = (
        "min_collision_distance_tf_preferred_m" in rows[0]
        and "min_collision_distance_fk_only_m" in rows[0]
    )
    if has_two_mode_cols:
        d_tf_mode = np.array([float(row["min_collision_distance_tf_preferred_m"]) for row in rows], dtype=np.float64)
        d_fk_mode = np.array([float(row["min_collision_distance_fk_only_m"]) for row in rows], dtype=np.float64)
    else:
        d_tf_mode = np.full_like(d_min, np.nan)
        d_fk_mode = np.full_like(d_min, np.nan)

    e_cmd_pos = np.array([float(row["cmd_goal_pos_err_m"]) for row in rows], dtype=np.float64)
    e_cmd_rot = np.array([float(row["cmd_goal_rot_err_rad"]) for row in rows], dtype=np.float64)
    e_pick_pos = np.array([float(row["goal_pick_pos_err_m"]) for row in rows], dtype=np.float64)
    e_place_pos = np.array([float(row["goal_place_pos_err_m"]) for row in rows], dtype=np.float64)
    e_min_pos = np.array([float(row["goal_min_pos_err_m"]) for row in rows], dtype=np.float64)
    e_pick_rot = np.array([float(row["goal_pick_rot_err_rad"]) for row in rows], dtype=np.float64)
    e_place_rot = np.array([float(row["goal_place_rot_err_rad"]) for row in rows], dtype=np.float64)
    e_min_rot = np.array([float(row["goal_min_rot_err_rad"]) for row in rows], dtype=np.float64)

    # Plot 1: collision distances over time
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(t, d_min, label="min collision distance", linewidth=1.8)
    if plot_mode_comparison and has_two_mode_cols and np.isfinite(d_tf_mode).any() and np.isfinite(d_fk_mode).any():
        ax1.plot(t, d_tf_mode, label="min distance (tf_preferred)", linewidth=1.6, linestyle="--")
        ax1.plot(t, d_fk_mode, label="min distance (fk_only)", linewidth=1.6, linestyle=":")
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1.0, label="contact boundary (0 m)")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("distance [m]")
    ax1.set_title("Minimum Robot-Obstacle Distance Over Time")
    ax1.grid(True, alpha=0.3)
    if plot_mode_comparison:
        ax1.legend(loc="best")
    p1 = output_csv.with_name(output_csv.stem + "_collision_distance.png")
    fig1.tight_layout()
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    p4 = None
    if plot_mode_comparison and has_two_mode_cols and np.isfinite(d_tf_mode).any() and np.isfinite(d_fk_mode).any():
        fig4, (ax5, ax6) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax5.plot(t, d_tf_mode, label="tf_preferred", linewidth=1.6)
        ax5.plot(t, d_fk_mode, label="fk_only", linewidth=1.6)
        ax5.axhline(0.0, color="red", linestyle="--", linewidth=1.0, label="contact boundary (0 m)")
        ax5.set_ylabel("distance [m]")
        ax5.set_title("Minimum Robot-Obstacle Distance: TF-Preferred vs FK-Only")
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc="best")

        ax6.plot(t, d_tf_mode - d_fk_mode, label="tf_preferred - fk_only", linewidth=1.4)
        ax6.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax6.set_xlabel("time [s]")
        ax6.set_ylabel("distance diff [m]")
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc="best")

        p4 = output_csv.with_name(output_csv.stem + "_collision_distance_modes.png")
        fig4.tight_layout()
        fig4.savefig(p4, dpi=160)
        plt.close(fig4)

    # Plot 2: goal errors over time
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax2.plot(t, e_cmd_pos, label="commanded goal pos err", linewidth=2.0)
    if include_static_goal_refs:
        ax2.plot(t, e_pick_pos, label="pick pos err (static ref)", linewidth=1.0, alpha=0.7)
        ax2.plot(t, e_place_pos, label="place pos err (static ref)", linewidth=1.0, alpha=0.7)
        ax2.plot(t, e_min_pos, label="min(pos err static refs)", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("position error [m]")
    ax2.set_title("End-Effector Goal Errors Over Time")
    ax2.grid(True, alpha=0.3)
    if include_static_goal_refs:
        ax2.legend(loc="best")

    ax3.plot(t, e_cmd_rot, label="commanded goal rot err", linewidth=2.0)
    if include_static_goal_refs:
        ax3.plot(t, e_pick_rot, label="pick rot err (static ref)", linewidth=1.0, alpha=0.7)
        ax3.plot(t, e_place_rot, label="place rot err (static ref)", linewidth=1.0, alpha=0.7)
        ax3.plot(t, e_min_rot, label="min(rot err static refs)", linewidth=1.2, alpha=0.7)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("orientation error [rad]")
    ax3.grid(True, alpha=0.3)
    if include_static_goal_refs:
        ax3.legend(loc="best")
    p2 = output_csv.with_name(output_csv.stem + "_goal_errors.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    # Optional Plot 3: closest obstacle frequency
    p3 = None
    if plot_closest_hist:
        closest_counts = Counter(str(row["closest_object_name"]) for row in rows)
        top_items = closest_counts.most_common(12)
    else:
        top_items = []
    if plot_closest_hist and top_items:
        labels = [name for name, _ in top_items]
        values = [count for _, count in top_items]
        fig3, ax4 = plt.subplots(figsize=(11, 5))
        ax4.bar(range(len(values)), values)
        ax4.set_xticks(range(len(values)))
        ax4.set_xticklabels(labels, rotation=35, ha="right")
        ax4.set_ylabel("sample count")
        ax4.set_title("Most Frequent Closest Obstacles")
        ax4.grid(True, axis="y", alpha=0.3)
        p3 = output_csv.with_name(output_csv.stem + "_closest_object_hist.png")
        fig3.tight_layout()
        fig3.savefig(p3, dpi=160)
        plt.close(fig3)

    print(f"[INFO] Saved plots:")
    print(f"       - {p1}")
    print(f"       - {p2}")
    if p3 is not None:
        print(f"       - {p3}")
    if p4 is not None:
        print(f"       - {p4}")


def _write_plot_csvs(rows: list[dict[str, object]], output_csv: Path) -> tuple[Path, Path]:
    dist_csv = output_csv.with_name(output_csv.stem + "_plot_collision_distance.csv")
    goal_csv = output_csv.with_name(output_csv.stem + "_plot_goal_errors.csv")

    dist_fields_pref = [
        "stamp_ns",
        "t_rel_s",
        "min_collision_distance_m",
        "collision_distance_mode",
        "min_collision_distance_fk_only_m",
        "min_collision_distance_tf_preferred_m",
    ]
    goal_fields_pref = [
        "stamp_ns",
        "t_rel_s",
        "cmd_goal_pos_err_m",
        "cmd_goal_rot_err_rad",
        "cmd_goal_available",
        "cmd_goal_age_s",
    ]

    dist_fields = [field for field in dist_fields_pref if field in rows[0]]
    goal_fields = [field for field in goal_fields_pref if field in rows[0]]

    with dist_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in dist_fields})

    with goal_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=goal_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in goal_fields})

    return dist_csv, goal_csv


def _latest_bag_dir(rosbags_dir: Path) -> Path:
    bag_dirs = [path for path in rosbags_dir.iterdir() if path.is_dir() and path.name.startswith("rosbag2_")]
    if not bag_dirs:
        raise FileNotFoundError(f"No rosbag folders found in {rosbags_dir}")
    return sorted(bag_dirs)[-1]


def _load_robot_spheres_specs(model: pin.Model, sphere_yaml_path: Path) -> list[RobotSphereSpec]:
    world = yaml.safe_load(sphere_yaml_path.read_text())
    coll = dict(world.get("collision_spheres", {}))
    frame_names = {frame.name: idx for idx, frame in enumerate(model.frames)}
    specs: list[RobotSphereSpec] = []
    missing_links: list[str] = []
    for storm_link_name, spheres in coll.items():
        ros_link_name = _robot_link_name_from_storm_link(str(storm_link_name))
        if ros_link_name not in frame_names:
            missing_links.append(ros_link_name)
            continue
        frame_id = frame_names[ros_link_name]
        for sphere in spheres:
            center = np.array(sphere["center"], dtype=np.float64)
            radius = float(sphere["radius"])
            specs.append(
                RobotSphereSpec(
                    frame_id=frame_id,
                    center_local=center,
                    radius=radius,
                    link_name=ros_link_name,
                )
            )
    if not specs:
        raise RuntimeError("No robot collision spheres could be loaded for this URDF.")
    if missing_links:
        preview = ", ".join(sorted(set(missing_links))[:8])
        print(f"[WARN] Missing sphere link frames in URDF (ignored): {preview}")
    return specs


def _build_static_obstacles_in_base(
    static_world: dict,
    tf_buffer: OfflineTFBuffer,
    base_frame: str,
) -> tuple[list[SphereObs], list[CubeObs]]:
    world_model = dict(static_world.get("world_model", {}))
    coll_objs = dict(world_model.get("coll_objs", {}))
    sphere_dict = dict(coll_objs.get("sphere", {}))
    cube_dict = dict(coll_objs.get("cube", {}))

    spheres: list[SphereObs] = []
    cubes: list[CubeObs] = []

    for name, spec in sphere_dict.items():
        frame = str(spec.get("frame", base_frame))
        position = np.array(spec["position"], dtype=np.float64)
        target_M_source = tf_buffer.lookup_transform(base_frame, frame)
        if target_M_source is None:
            continue
        center = target_M_source.translation + target_M_source.rotation @ position
        spheres.append(SphereObs(name=f"static_sphere:{name}", center=center, radius=float(spec["radius"]), source="static"))

    for name, spec in cube_dict.items():
        frame = str(spec.get("frame", base_frame))
        pose_xyzquat = np.array(spec["pose"], dtype=np.float64)
        source_M_obj = pin.XYZQUATToSE3(pose_xyzquat)
        target_M_source = tf_buffer.lookup_transform(base_frame, frame)
        if target_M_source is None:
            continue
        base_M_obj = target_M_source * source_M_obj
        cubes.append(
            CubeObs(
                name=f"static_cube:{name}",
                center=np.array(base_M_obj.translation, dtype=np.float64),
                rotation=np.array(base_M_obj.rotation, dtype=np.float64),
                dims=np.array(spec["dims"], dtype=np.float64),
                source="static",
            )
        )
    return spheres, cubes


def _extract_raw_markers(marker_array_msg) -> list[RawMarkerObstacle]:
    raw: list[RawMarkerObstacle] = []
    for marker in marker_array_msg.markers:
        if marker.action in (marker.DELETE, marker.DELETEALL):
            continue
        marker_type = int(marker.type)
        if marker_type not in (marker.CYLINDER, marker.CUBE, marker.SPHERE):
            continue
        frame = str(marker.header.frame_id)
        name = f"{marker.ns}:{int(marker.id)}" if marker.ns else f"id:{int(marker.id)}"
        raw.append(
            RawMarkerObstacle(
                name=name,
                source_frame=frame,
                marker_type=marker_type,
                pose_xyzquat=_pose_to_xyzquat(marker.pose),
                scale_xyz=np.array([marker.scale.x, marker.scale.y, marker.scale.z], dtype=np.float64),
            )
        )
    return raw


def _raw_markers_to_base(
    raw_markers: list[RawMarkerObstacle],
    tf_buffer: OfflineTFBuffer,
    base_frame: str,
) -> tuple[list[SphereObs], list[CubeObs], list[CylinderObs]]:
    spheres: list[SphereObs] = []
    cubes: list[CubeObs] = []
    cylinders: list[CylinderObs] = []
    for raw in raw_markers:
        source_M_obj = pin.XYZQUATToSE3(raw.pose_xyzquat)
        source_frame = raw.source_frame if raw.source_frame else base_frame
        target_M_source = tf_buffer.lookup_transform(base_frame, source_frame)
        if target_M_source is None:
            continue
        base_M_obj = target_M_source * source_M_obj

        if raw.marker_type == 2:  # SPHERE
            radius = 0.5 * float(max(raw.scale_xyz))
            if radius <= 0.0:
                continue
            spheres.append(
                SphereObs(
                    name=f"marker_sphere:{raw.name}",
                    center=np.array(base_M_obj.translation, dtype=np.float64),
                    radius=radius,
                    source="dynamic_marker",
                )
            )
        elif raw.marker_type == 1:  # CUBE
            dims = np.array(raw.scale_xyz, dtype=np.float64)
            if np.min(dims) <= 0.0:
                continue
            cubes.append(
                CubeObs(
                    name=f"marker_cube:{raw.name}",
                    center=np.array(base_M_obj.translation, dtype=np.float64),
                    rotation=np.array(base_M_obj.rotation, dtype=np.float64),
                    dims=dims,
                    source="dynamic_marker",
                )
            )
        elif raw.marker_type == 3:  # CYLINDER
            radius = 0.5 * float(max(raw.scale_xyz[0], raw.scale_xyz[1]))
            height = float(raw.scale_xyz[2])
            if radius <= 0.0 or height <= 0.0:
                continue
            cylinders.append(
                CylinderObs(
                    name=f"marker_cylinder:{raw.name}",
                    center=np.array(base_M_obj.translation, dtype=np.float64),
                    rotation=np.array(base_M_obj.rotation, dtype=np.float64),
                    radius=radius,
                    height=height,
                    source="dynamic_marker",
                )
            )
    return spheres, cubes, cylinders


def _extract_raw_pose_spheres(pose_array_msg, *, radius: float) -> list[RawPoseSphere]:
    source_frame = str(pose_array_msg.header.frame_id)
    raw: list[RawPoseSphere] = []
    for idx, pose in enumerate(pose_array_msg.poses):
        raw.append(
            RawPoseSphere(
                name=f"pose_{idx}",
                source_frame=source_frame,
                center=np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
                radius=radius,
            )
        )
    return raw


def _raw_pose_spheres_to_base(
    raw_pose_spheres: list[RawPoseSphere],
    tf_buffer: OfflineTFBuffer,
    base_frame: str,
) -> list[SphereObs]:
    spheres: list[SphereObs] = []
    for raw in raw_pose_spheres:
        source_frame = raw.source_frame if raw.source_frame else base_frame
        target_M_source = tf_buffer.lookup_transform(base_frame, source_frame)
        if target_M_source is None:
            continue
        center = target_M_source.translation + target_M_source.rotation @ raw.center
        spheres.append(
            SphereObs(
                name=f"pose_sphere:{raw.name}",
                center=center,
                radius=float(raw.radius),
                source="dynamic_pose",
            )
        )
    return spheres


def _compute_min_clearance(
    robot_spheres: list[tuple[np.ndarray, float, str]],
    sphere_obs: list[SphereObs],
    cube_obs: list[CubeObs],
    cyl_obs: list[CylinderObs],
) -> tuple[float, str, str, float, float]:
    global_min = float("inf")
    min_name = ""
    min_type = ""
    min_static = float("inf")
    min_dynamic = float("inf")

    for center, radius, _ in robot_spheres:
        for obs in sphere_obs:
            clearance = float(np.linalg.norm(center - obs.center)) - (radius + obs.radius)
            if clearance < global_min:
                global_min = clearance
                min_name = obs.name
                min_type = "sphere"
            if obs.source == "static":
                min_static = min(min_static, clearance)
            else:
                min_dynamic = min(min_dynamic, clearance)

        for obs in cube_obs:
            point_obj = obs.rotation.T @ (center - obs.center)
            clearance = _sdf_box(point_obj, obs.dims) - radius
            if clearance < global_min:
                global_min = clearance
                min_name = obs.name
                min_type = "cube"
            if obs.source == "static":
                min_static = min(min_static, clearance)
            else:
                min_dynamic = min(min_dynamic, clearance)

        for obs in cyl_obs:
            point_obj = obs.rotation.T @ (center - obs.center)
            clearance = _sdf_cylinder(point_obj, obs.radius, obs.height) - radius
            if clearance < global_min:
                global_min = clearance
                min_name = obs.name
                min_type = "cylinder"
            if obs.source == "static":
                min_static = min(min_static, clearance)
            else:
                min_dynamic = min(min_dynamic, clearance)

    return global_min, min_name, min_type, min_static, min_dynamic


def _compute_robot_spheres_in_base(
    *,
    robot_sphere_specs: list[RobotSphereSpec],
    tf_buffer: OfflineTFBuffer,
    base_frame: str,
    data: pin.Data,
    world_M_base: pin.SE3,
) -> tuple[list[tuple[np.ndarray, float, str]], int, int]:
    base_M_world = world_M_base.inverse()
    robot_spheres: list[tuple[np.ndarray, float, str]] = []
    tf_count = 0
    fk_count = 0
    for spec in robot_sphere_specs:
        base_M_link_tf = tf_buffer.lookup_transform(base_frame, spec.link_name)
        if base_M_link_tf is not None:
            center = base_M_link_tf.translation + base_M_link_tf.rotation @ spec.center_local
            tf_count += 1
        else:
            world_M_link = data.oMf[spec.frame_id]
            base_M_link = base_M_world * world_M_link
            center = base_M_link.translation + base_M_link.rotation @ spec.center_local
            fk_count += 1
        robot_spheres.append((center, spec.radius, spec.link_name))
    return robot_spheres, tf_count, fk_count


def _compute_robot_spheres_fk_only(
    *,
    robot_sphere_specs: list[RobotSphereSpec],
    data: pin.Data,
    world_M_base: pin.SE3,
) -> list[tuple[np.ndarray, float, str]]:
    base_M_world = world_M_base.inverse()
    robot_spheres: list[tuple[np.ndarray, float, str]] = []
    for spec in robot_sphere_specs:
        world_M_link = data.oMf[spec.frame_id]
        base_M_link = base_M_world * world_M_link
        center = base_M_link.translation + base_M_link.rotation @ spec.center_local
        robot_spheres.append((center, spec.radius, spec.link_name))
    return robot_spheres


def _choose_joint_topic(topic_types: dict[str, str]) -> tuple[str, str]:
    candidates = [
        ("/rtcosmik/q", "std_msgs/msg/Float64MultiArray"),
        ("/joint_states", "sensor_msgs/msg/JointState"),
        ("/franka/joint_states", "sensor_msgs/msg/JointState"),
        ("/rtcosmik/joint_states", "sensor_msgs/msg/JointState"),
    ]
    for topic, msg_type in candidates:
        if topic in topic_types and topic_types[topic] == msg_type:
            return topic, msg_type
    raise RuntimeError("No supported joint topic found in bag.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute collision-distance and goal-error metrics from a ROS2 bag.")
    parser.add_argument(
        "--bag",
        type=Path,
        default=None,
        help="Path to bag directory (containing metadata.yaml). Default: latest under /workspace/ros2_ws/rosbags",
    )
    parser.add_argument(
        "--rosbags-dir",
        type=Path,
        default=Path("/workspace/ros2_ws/rosbags"),
        help="Directory containing rosbag folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <bag_dir>/collision_goal_metrics.csv",
    )
    parser.add_argument("--base-frame", type=str, default="fer_link0")
    parser.add_argument("--ee-frame", type=str, default="fer_hand_tcp")
    parser.add_argument(
        "--goal-topic",
        type=str,
        default="/goal_pose",
        help="PoseStamped topic with commanded end-effector goals.",
    )
    parser.add_argument("--collision-pose-radius", type=float, default=0.05)
    parser.add_argument(
        "--robot-sphere-yaml",
        type=Path,
        default=Path("/workspace/storm/content/configs/robot/franka.yml"),
    )
    parser.add_argument(
        "--fallback-urdf",
        type=Path,
        default=Path("/workspace/storm/content/assets/urdf/franka_description/franka_panda_no_gripper.urdf"),
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep one row every N joint samples (N>=1).",
    )
    parser.add_argument(
        "--collision-distance-mode",
        type=str,
        default="fk_only",
        choices=["tf_preferred", "fk_only", "both"],
        help=(
            "Robot sphere source for collision distance eval: "
            "tf_preferred (current), fk_only (legacy), or both (compute and output both)."
        ),
    )
    parser.add_argument(
        "--plot-mode-comparison",
        action="store_true",
        help="Also generate the TF-vs-FK mode comparison collision plot (requires --collision-distance-mode both).",
    )
    parser.add_argument(
        "--plot-closest-hist",
        action="store_true",
        help="Also generate closest-obstacle histogram plot.",
    )
    parser.add_argument(
        "--plot-static-goal-refs",
        action="store_true",
        help="Also overlay static pick/place reference errors on the goal-error plot.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving PNG plots.",
    )
    args = parser.parse_args()

    if args.bag is None:
        bag_dir = _latest_bag_dir(args.rosbags_dir)
    else:
        bag_dir = args.bag
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag directory does not exist: {bag_dir}")

    output_csv = args.output_csv or (bag_dir / "collision_goal_metrics.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {info.name: info.type for info in reader.get_all_topics_and_types()}

    joint_topic, joint_topic_type = _choose_joint_topic(topic_types)
    print(f"[INFO] Using bag: {bag_dir}")
    print(f"[INFO] Joint topic: {joint_topic} ({joint_topic_type})")

    goal_topic = str(args.goal_topic)
    goal_topic_type = topic_types.get(goal_topic)
    goal_topic_enabled = goal_topic_type == "geometry_msgs/msg/PoseStamped"
    if goal_topic_type is None:
        print(f"[WARN] Goal topic '{goal_topic}' not found in bag. Commanded-goal error columns will be NaN.")
    elif not goal_topic_enabled:
        print(
            f"[WARN] Goal topic '{goal_topic}' has type '{goal_topic_type}', expected "
            "'geometry_msgs/msg/PoseStamped'. Commanded-goal error columns will be NaN."
        )

    parse_topics = {
        "/tf_static",
        "/tf",
        "/robot_description_with_collision",
        "/robot_description",
        "/rtcosmik/collision_markers",
        "/rtcosmik/collision_poses",
        joint_topic,
    }
    if goal_topic_enabled:
        parse_topics.add(goal_topic)
    parse_topics = {topic for topic in parse_topics if topic in topic_types}
    msg_classes = {topic: get_message(topic_types[topic]) for topic in parse_topics}

    tf_buffer = OfflineTFBuffer()
    static_world = _default_static_world()
    static_spheres: list[SphereObs] = []
    static_cubes: list[CubeObs] = []

    latest_raw_markers: list[RawMarkerObstacle] = []
    latest_raw_pose_spheres: list[RawPoseSphere] = []
    latest_goal_base: Optional[pin.SE3] = None
    latest_goal_stamp_ns: Optional[int] = None

    urdf_xml: Optional[str] = None
    model: Optional[pin.Model] = None
    data: Optional[pin.Data] = None
    ee_frame_id: Optional[int] = None
    base_frame_id: Optional[int] = None
    q_nominal: Optional[np.ndarray] = None
    arm_joint_to_q_idx: dict[str, int] = {}
    robot_sphere_specs: list[RobotSphereSpec] = []

    pick_goal = pin.XYZQUATToSE3(np.array([0.32, 0.0, 0.235, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    place_goal = pin.XYZQUATToSE3(np.array([0.63, 0.0, 0.24, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))

    rows: list[dict[str, object]] = []
    first_stamp_ns: Optional[int] = None
    joint_counter = 0

    while reader.has_next():
        topic, raw_data, bag_time_ns = reader.read_next()
        if topic not in parse_topics:
            continue
        msg = deserialize_message(raw_data, msg_classes[topic])
        stamp_ns = int(bag_time_ns)
        if first_stamp_ns is None:
            first_stamp_ns = stamp_ns

        if topic == "/tf_static":
            tf_buffer.update_tf_message(msg, is_static=True, bag_time_ns=stamp_ns)
            static_spheres, static_cubes = _build_static_obstacles_in_base(static_world, tf_buffer, args.base_frame)
            continue
        if topic == "/tf":
            tf_buffer.update_tf_message(msg, is_static=False, bag_time_ns=stamp_ns)
            continue
        if topic in ("/robot_description_with_collision", "/robot_description"):
            if urdf_xml is None and getattr(msg, "data", ""):
                urdf_xml = str(msg.data)
                try:
                    model = pin.buildModelFromXML(urdf_xml)
                    data = model.createData()
                    frame_names = {frame.name: idx for idx, frame in enumerate(model.frames)}
                    if args.ee_frame not in frame_names:
                        raise RuntimeError(f"EE frame '{args.ee_frame}' is not present in bag URDF.")
                    if args.base_frame not in frame_names:
                        raise RuntimeError(f"Base frame '{args.base_frame}' is not present in bag URDF.")
                    ee_frame_id = frame_names[args.ee_frame]
                    base_frame_id = frame_names[args.base_frame]
                    q_nominal = pin.neutral(model)
                    for joint_name in [f"fer_joint{i}" for i in range(1, 8)]:
                        joint_id = model.getJointId(joint_name)
                        if joint_id == 0:
                            continue
                        q_idx = model.idx_qs[joint_id]
                        arm_joint_to_q_idx[joint_name] = int(q_idx)
                    robot_sphere_specs = _load_robot_spheres_specs(model, args.robot_sphere_yaml)
                    print(f"[INFO] Loaded robot model from bag URDF with {model.nq} nq, {model.nv} nv.")
                except Exception as exc:
                    print(f"[WARN] Failed to build model from bag URDF: {exc}. Falling back to {args.fallback_urdf}")
                    model = pin.buildModelFromUrdf(str(args.fallback_urdf))
                    data = model.createData()
                    frame_names = {frame.name: idx for idx, frame in enumerate(model.frames)}
                    ee_frame_id = frame_names.get(args.ee_frame)
                    base_frame_id = frame_names.get(args.base_frame)
                    if ee_frame_id is None or base_frame_id is None:
                        raise RuntimeError(
                            f"Fallback URDF missing required frames base='{args.base_frame}' ee='{args.ee_frame}'."
                        )
                    q_nominal = pin.neutral(model)
                    for i in range(1, 8):
                        panda_name = f"panda_joint{i}"
                        joint_id = model.getJointId(panda_name)
                        if joint_id == 0:
                            continue
                        q_idx = model.idx_qs[joint_id]
                        arm_joint_to_q_idx[f"fer_joint{i}"] = int(q_idx)
                    robot_sphere_specs = _load_robot_spheres_specs(model, args.robot_sphere_yaml)
            continue
        if topic == goal_topic:
            source_frame = str(msg.header.frame_id).strip() or args.base_frame
            source_M_goal = pin.XYZQUATToSE3(_pose_to_xyzquat(msg.pose))
            if source_frame == args.base_frame:
                latest_goal_base = source_M_goal
                latest_goal_stamp_ns = stamp_ns
            else:
                base_M_source = tf_buffer.lookup_transform(args.base_frame, source_frame)
                if base_M_source is not None:
                    latest_goal_base = base_M_source * source_M_goal
                    latest_goal_stamp_ns = stamp_ns
            continue
        if topic == "/rtcosmik/collision_markers":
            latest_raw_markers = _extract_raw_markers(msg)
            continue
        if topic == "/rtcosmik/collision_poses":
            latest_raw_pose_spheres = _extract_raw_pose_spheres(msg, radius=float(args.collision_pose_radius))
            continue
        if topic == joint_topic:
            if model is None or data is None or ee_frame_id is None or base_frame_id is None or q_nominal is None:
                continue
            joint_counter += 1
            if int(args.sample_every) > 1 and (joint_counter % int(args.sample_every)) != 0:
                continue

            q = np.array(q_nominal, dtype=np.float64, copy=True)
            if joint_topic_type == "std_msgs/msg/Float64MultiArray":
                values = list(msg.data)
                if len(values) < 7:
                    continue
                for idx in range(7):
                    joint_name = f"fer_joint{idx + 1}"
                    if joint_name in arm_joint_to_q_idx:
                        q[arm_joint_to_q_idx[joint_name]] = float(values[idx])
            else:
                name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
                missing = False
                for i in range(1, 8):
                    joint_name = f"fer_joint{i}"
                    if joint_name not in name_to_pos:
                        missing = True
                        break
                    if joint_name in arm_joint_to_q_idx:
                        q[arm_joint_to_q_idx[joint_name]] = float(name_to_pos[joint_name])
                if missing:
                    continue

            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            world_M_base = data.oMf[base_frame_id]
            collision_distance_mode = str(args.collision_distance_mode)
            robot_spheres_tf: list[tuple[np.ndarray, float, str]] = []
            robot_spheres_fk: list[tuple[np.ndarray, float, str]] = []
            robot_spheres_tf_count = 0
            robot_spheres_fk_count = 0
            if collision_distance_mode in ("tf_preferred", "both"):
                robot_spheres_tf, robot_spheres_tf_count, robot_spheres_fk_count = _compute_robot_spheres_in_base(
                    robot_sphere_specs=robot_sphere_specs,
                    tf_buffer=tf_buffer,
                    base_frame=args.base_frame,
                    data=data,
                    world_M_base=world_M_base,
                )
            if collision_distance_mode in ("fk_only", "both"):
                robot_spheres_fk = _compute_robot_spheres_fk_only(
                    robot_sphere_specs=robot_sphere_specs,
                    data=data,
                    world_M_base=world_M_base,
                )

            marker_spheres, marker_cubes, marker_cyls = _raw_markers_to_base(
                latest_raw_markers, tf_buffer, args.base_frame
            )
            pose_spheres = _raw_pose_spheres_to_base(latest_raw_pose_spheres, tf_buffer, args.base_frame)

            # Requested eval mode: consider ONLY dynamic marker cylinders as obstacles.
            # Ignore static world cubes/spheres, pose-array spheres, and non-cylinder marker shapes.
            _ = marker_spheres, marker_cubes, pose_spheres, static_spheres, static_cubes
            all_spheres: list[SphereObs] = []
            all_cubes: list[CubeObs] = []
            all_cyls = list(marker_cyls)

            min_dist_tf = float("nan")
            min_name_tf = ""
            min_type_tf = ""
            min_static_tf = float("nan")
            min_dynamic_tf = float("nan")
            if collision_distance_mode in ("tf_preferred", "both"):
                min_dist_tf, min_name_tf, min_type_tf, min_static_tf, min_dynamic_tf = _compute_min_clearance(
                    robot_spheres_tf,
                    all_spheres,
                    all_cubes,
                    all_cyls,
                )

            min_dist_fk = float("nan")
            min_name_fk = ""
            min_type_fk = ""
            min_static_fk = float("nan")
            min_dynamic_fk = float("nan")
            if collision_distance_mode in ("fk_only", "both"):
                min_dist_fk, min_name_fk, min_type_fk, min_static_fk, min_dynamic_fk = _compute_min_clearance(
                    robot_spheres_fk,
                    all_spheres,
                    all_cubes,
                    all_cyls,
                )

            if collision_distance_mode == "fk_only":
                min_dist = min_dist_fk
                min_name = min_name_fk
                min_type = min_type_fk
                min_static_dist = min_static_fk
                min_dynamic_dist = min_dynamic_fk
            else:
                min_dist = min_dist_tf
                min_name = min_name_tf
                min_type = min_type_tf
                min_static_dist = min_static_tf
                min_dynamic_dist = min_dynamic_tf

            base_M_ee_tf = tf_buffer.lookup_transform(args.base_frame, args.ee_frame)
            if base_M_ee_tf is not None:
                ee_pos = np.array(base_M_ee_tf.translation, dtype=np.float64)
                ee_rot = np.array(base_M_ee_tf.rotation, dtype=np.float64)
                ee_pose_source = "tf"
            else:
                base_M_world = world_M_base.inverse()
                world_M_ee = data.oMf[ee_frame_id]
                base_M_ee = base_M_world * world_M_ee
                ee_pos = np.array(base_M_ee.translation, dtype=np.float64)
                ee_rot = np.array(base_M_ee.rotation, dtype=np.float64)
                ee_pose_source = "fk"

            pick_pos_err = float(np.linalg.norm(ee_pos - pick_goal.translation))
            place_pos_err = float(np.linalg.norm(ee_pos - place_goal.translation))
            pick_rot_err = _quat_rot_err_rad(np.array(pick_goal.rotation, dtype=np.float64), ee_rot)
            place_rot_err = _quat_rot_err_rad(np.array(place_goal.rotation, dtype=np.float64), ee_rot)
            if latest_goal_base is None:
                cmd_goal_pos_err = float("nan")
                cmd_goal_rot_err = float("nan")
                cmd_goal_age_s = float("nan")
                cmd_goal_available = 0
            else:
                cmd_goal_pos_err = float(np.linalg.norm(ee_pos - latest_goal_base.translation))
                cmd_goal_rot_err = _quat_rot_err_rad(np.array(latest_goal_base.rotation, dtype=np.float64), ee_rot)
                if latest_goal_stamp_ns is None:
                    cmd_goal_age_s = float("nan")
                else:
                    cmd_goal_age_s = float((stamp_ns - latest_goal_stamp_ns) * 1.0e-9)
                cmd_goal_available = 1

            rows.append(
                {
                    "stamp_ns": stamp_ns,
                    "t_rel_s": (stamp_ns - int(first_stamp_ns)) * 1.0e-9,
                    "min_collision_distance_m": min_dist,
                    "closest_object_name": min_name,
                    "closest_object_type": min_type,
                    "min_static_collision_distance_m": min_static_dist,
                    "min_dynamic_collision_distance_m": min_dynamic_dist,
                    "collision_distance_mode": collision_distance_mode,
                    "min_collision_distance_tf_preferred_m": min_dist_tf,
                    "min_collision_distance_fk_only_m": min_dist_fk,
                    "closest_object_tf_preferred_name": min_name_tf,
                    "closest_object_fk_only_name": min_name_fk,
                    "closest_object_tf_preferred_type": min_type_tf,
                    "closest_object_fk_only_type": min_type_fk,
                    "num_static_objects": 0,
                    "num_dynamic_objects": len(marker_cyls),
                    "goal_pick_pos_err_m": pick_pos_err,
                    "goal_pick_rot_err_rad": pick_rot_err,
                    "goal_place_pos_err_m": place_pos_err,
                    "goal_place_rot_err_rad": place_rot_err,
                    "goal_min_pos_err_m": min(pick_pos_err, place_pos_err),
                    "goal_min_rot_err_rad": min(pick_rot_err, place_rot_err),
                    "cmd_goal_pos_err_m": cmd_goal_pos_err,
                    "cmd_goal_rot_err_rad": cmd_goal_rot_err,
                    "cmd_goal_age_s": cmd_goal_age_s,
                    "cmd_goal_available": cmd_goal_available,
                    "ee_pose_source": ee_pose_source,
                    "robot_spheres_tf_count": robot_spheres_tf_count,
                    "robot_spheres_fk_count": robot_spheres_fk_count,
                }
            )

    if not rows:
        raise RuntimeError("No rows were computed. Check that bag topics and robot model frames match.")

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    dist_plot_csv, goal_plot_csv = _write_plot_csvs(rows, output_csv)

    if not args.no_plots:
        _save_plots(
            rows,
            output_csv,
            plot_mode_comparison=bool(args.plot_mode_comparison),
            plot_closest_hist=bool(args.plot_closest_hist),
            include_static_goal_refs=bool(args.plot_static_goal_refs),
        )

    min_dist_val = min(float(row["min_collision_distance_m"]) for row in rows)
    collision_distance_mode = str(args.collision_distance_mode)
    if collision_distance_mode in ("tf_preferred", "both"):
        tf_sphere_samples = sum(int(row.get("robot_spheres_tf_count", 0)) for row in rows)
        fk_sphere_samples = sum(int(row.get("robot_spheres_fk_count", 0)) for row in rows)
        total_sphere_samples = tf_sphere_samples + fk_sphere_samples
        if total_sphere_samples > 0:
            fk_ratio = fk_sphere_samples / float(total_sphere_samples)
            print(
                f"[INFO] Robot sphere placement samples (tf_preferred mode): TF={tf_sphere_samples}, "
                f"FK-fallback={fk_sphere_samples} ({100.0 * fk_ratio:.2f}%)."
            )
            if fk_sphere_samples > 0:
                print(
                    f"[WARN] Robot sphere placement used FK fallback for {100.0 * fk_ratio:.2f}% "
                    "of sphere samples. Collision distances may be less consistent than TF-based evaluation."
                )
    if collision_distance_mode == "both":
        tf_vals = [float(row["min_collision_distance_tf_preferred_m"]) for row in rows]
        fk_vals = [float(row["min_collision_distance_fk_only_m"]) for row in rows]
        min_dist_tf = min(v for v in tf_vals if math.isfinite(v))
        min_dist_fk = min(v for v in fk_vals if math.isfinite(v))
        print(f"[INFO] Global minimum collision distance (tf_preferred): {min_dist_tf:.6f} m")
        print(f"[INFO] Global minimum collision distance (fk_only): {min_dist_fk:.6f} m")
    print(f"[INFO] Wrote {len(rows)} rows: {output_csv}")
    print(f"[INFO] Wrote plot CSV (collision distance): {dist_plot_csv}")
    print(f"[INFO] Wrote plot CSV (goal errors): {goal_plot_csv}")
    print(f"[INFO] Global minimum collision distance: {min_dist_val:.6f} m")


if __name__ == "__main__":
    main()
