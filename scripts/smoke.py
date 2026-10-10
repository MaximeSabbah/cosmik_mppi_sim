#!/usr/bin/env python3
"""Run and monitor only this simulation's process group; write evidence as JSON.

Use via smoke.sh. Arrival counts require both configured position and rotation
tolerances; timeout alternations alone cannot pass the reaching check.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
from linear_feedback_controller_msgs.msg import Control
from tf2_ros import Buffer, TransformListener, TransformException


class Monitor(Node):
    def __init__(self):
        super().__init__('handoff_smoke_monitor')
        self.counts = Counter()
        self.sim_time = 0.0
        self.first_goal_time = None
        self.goal = None
        self.target = None
        self.target_order = []
        self.arrivals = Counter()
        self.visit_reached = False
        self.minimum = {'A': float('inf'), 'B': float('inf')}
        self.nonfinite = False
        self.joint_min = None
        self.joint_max = None
        self.human_x = []
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joints_cb, qos_profile_sensor_data)
        self.create_subscription(Control, '/control', self.control_cb, qos_profile_sensor_data)
        self.create_subscription(MarkerArray, '/rtcosmik/collision_markers', self.human_cb, 1)
        self.create_timer(0.05, self.sample)

    def clock_cb(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.counts['clock'] += 1

    def goal_cb(self, msg):
        self.goal = msg.pose
        self.counts['goals'] += 1
        if self.first_goal_time is None:
            self.first_goal_time = self.sim_time
        target = 'A' if msg.pose.position.y < 0 else 'B'
        if target != self.target:
            self.target = target
            self.target_order.append({'target': target, 'sim_time': self.sim_time})
            self.visit_reached = False

    def joints_cb(self, msg):
        self.counts['joint_states'] += 1
        values = np.asarray(msg.position)
        self.nonfinite |= not np.all(np.isfinite(values))
        by_name = dict(zip(msg.name, msg.position))
        if all(f'fer_joint{i}' in by_name for i in range(1, 8)):
            q = np.array([by_name[f'fer_joint{i}'] for i in range(1, 8)])
            self.joint_min = q if self.joint_min is None else np.minimum(self.joint_min, q)
            self.joint_max = q if self.joint_max is None else np.maximum(self.joint_max, q)

    def control_cb(self, msg):
        self.counts['control'] += 1
        values = [*msg.feedforward.data, *msg.feedback_gain.data, *msg.initial_state.joint_state.position]
        self.nonfinite |= not np.all(np.isfinite(values))

    def human_cb(self, msg):
        self.counts['human_frames'] += 1
        if msg.markers:
            self.human_x.append(msg.markers[0].pose.position.x)

    def sample(self):
        if self.goal is None:
            return
        try:
            tf = self.tf.lookup_transform('fer_link0', 'fer_hand_tcp', Time()).transform
        except TransformException:
            return
        self.counts['ee_samples'] += 1
        p, g = tf.translation, self.goal.position
        err = math.sqrt((p.x-g.x)**2 + (p.y-g.y)**2 + (p.z-g.z)**2)
        self.minimum[self.target] = min(self.minimum[self.target], err)
        q, r = tf.rotation, self.goal.orientation
        qa = np.array([q.x, q.y, q.z, q.w]); ra = np.array([r.x, r.y, r.z, r.w])
        cosine = abs(float(qa @ ra)) / (np.linalg.norm(qa) * np.linalg.norm(ra))
        angle = 2 * math.acos(float(np.clip(cosine, 0, 1)))
        if err < 0.06 and angle < 2.0 and not self.visit_reached:
            self.arrivals[self.target] += 1
            self.visit_reached = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', choices=['true', 'false'], default='true')
    parser.add_argument('--seconds', type=float, default=50)
    parser.add_argument('--wall-limit', type=float, default=240)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Monitor()
    started = time.monotonic()
    with (args.output/'launch.log').open('w') as log:
        proc = subprocess.Popen([str(Path(__file__).with_name('run.sh')), f'synthetic_human:={args.human}'],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while time.monotonic() - started < args.wall_limit and proc.poll() is None:
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.first_goal_time is not None and node.sim_time-node.first_goal_time >= args.seconds:
                    break
        finally:
            # Only terminate the process group that this script created.
            for sig, wait in [(signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 2)]:
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    continue
                # launch may exit before a spawned child; follow up on the group.
                if sig == signal.SIGKILL:
                    break
                time.sleep(0.5)
    log_text = (args.output/'launch.log').read_text(errors='replace')
    reached = node.arrivals['A'] > 0 and node.arrivals['B'] > 0
    live = node.counts['control'] > 100 and node.counts['ee_samples'] > 100
    duration = 0.0 if node.first_goal_time is None else node.sim_time-node.first_goal_time
    fixture_ok = args.human == 'false' or (len(node.human_x) > 100 and np.ptp(node.human_x) > 0.1)
    dynamic_updates = log_text.count('Updated dynamic collision spheres from live marker/pose inputs.')
    ingestion_ok = args.human == 'false' or dynamic_updates > 1
    profile_loaded = ("Loaded STORM task config: storm_task_file='franka_reacher_cat_real.yml'" in log_text
                      and 'mppi.cat_primitive_collision.enabled=True' in log_text)
    result = {
        'pass': bool(reached and live and fixture_ok and ingestion_ok and profile_loaded
                     and not node.nonfinite and duration >= args.seconds),
        'scope': 'Simulation transport and reaching smoke check; does not certify collision avoidance or prediction.',
        'human': args.human, 'wall_seconds': time.monotonic()-started,
        'cat_profile_loaded': profile_loaded, 'dynamic_geometry_update_logs': dynamic_updates,
        'simulated_seconds_after_first_goal': duration,
        'messages': dict(node.counts), 'target_visits': node.target_order,
        'spatial_arrivals': dict(node.arrivals),
        'minimum_target_error_m': {key: value if math.isfinite(value) else None for key, value in node.minimum.items()},
        'timeout_events': log_text.count('[task-event] target-timeout('),
        'nonfinite_state_or_command': bool(node.nonfinite),
        'joint_excursion_rad': None if node.joint_min is None else (node.joint_max-node.joint_min).tolist(),
        'task_events': [line for line in log_text.splitlines() if '[task-event]' in line],
    }
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'task_events'}, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
