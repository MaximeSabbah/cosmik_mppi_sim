#!/usr/bin/env python3
"""Check the active environment, package isolation, CUDA and baseline hashes."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['DEMO_ROOT']).resolve()
errors = []
report = {'python': sys.version, 'root': str(root), 'domain': os.environ['ROS_DOMAIN_ID'], 'packages': {}}
for name in ['torch', 'numpy', 'scipy', 'yaml', 'pinocchio', 'trimesh', 'urdf_parser_py',
             'storm_kit', 'storm_kit.mpc.task.reacher_task', 'storm_agimus_bridge.storm_mppi_planner_node',
             'storm_pick_place_demo.pick_place_task_manager_node', 'linear_feedback_controller_msgs_py']:
    try:
        mod = importlib.import_module(name)
        report['packages'][name] = {'path': str(mod.__file__), 'version': str(getattr(mod, '__version__', ''))}
        if name.startswith(('storm_', 'linear_feedback_controller_msgs_py')) and not Path(mod.__file__).resolve().is_relative_to(root):
            errors.append(f'{name} resolves outside the handoff')
    except Exception as exc:
        errors.append(f'{name}: {exc}')
try:
    import torch
    if not torch.cuda.is_available():
        errors.append('CUDA is unavailable; this profile expects an NVIDIA GPU')
    else:
        report['gpu'] = torch.cuda.get_device_name(0)
        report['cuda'] = torch.version.cuda
        torch.ones(1, device='cuda').add_(1).cpu()
except Exception as exc:
    errors.append(str(exc))
from ament_index_python.packages import get_package_prefix
for name in ['franka_description', 'franka_ign_ros2_control', 'linear_feedback_controller', 'agimus_demos_common']:
    try:
        prefix = Path(get_package_prefix(name))
        report['packages'][name] = {'prefix': str(prefix)}
        if not prefix.is_relative_to(root):
            errors.append(f'{name} comes from an unrelated ROS overlay: {prefix}')
    except Exception as exc:
        errors.append(str(exc))
baseline = json.loads((root/'provenance/baseline-configs.json').read_text())
report['baseline_config_changes'] = [name for name, expected in baseline['file_sha256'].items()
                                    if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected]
report['errors'] = errors
report['pass'] = not errors
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors))
