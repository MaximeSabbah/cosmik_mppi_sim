#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import os, sys
import traceback
import json
import time
import copy

#import numpy as np
import torch
import tempfile


from ..utils.torch_utils import find_first_idx, find_last_idx
#import multiprocessing import Queue
from torch.multiprocessing import Pool, Process, set_start_method, Queue

import numpy as np

class ControlProcess(object):
    def __init__(self, controller, control_space='acc', control_dt=0.01, exact_latency_prediction=False):
        self.exact_latency_prediction = bool(exact_latency_prediction)
        try:
            controller.rollout_fn.dynamics_model.robot_model.delete_lxml_objects()
        except Exception:
            pass

        # Persist controller to a temporary file so the optimization process can load it.
        # Using a temp file avoids relying on the current working directory being writable
        # (e.g., when running under ROS 2 launch).
        fd, control_path = tempfile.mkstemp(prefix="storm_control_instance_", suffix=".p")
        os.close(fd)
        self._control_cache_path = control_path
        torch.save(controller, control_path)
        
        try:
            controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()
        except Exception:
            pass
        self.command = None
        self.current_state = None
        self.done = False
        self.opt_data = 0.0
        self.n_dofs = controller.rollout_fn.dynamics_model.n_dofs
        
        self.traj_tstep = copy.deepcopy(controller.rollout_fn.dynamics_model._traj_tstep.detach().cpu())
        self.command_tstep = self.traj_tstep
        self.mpc_dt = 0.0 #None
        self.params = None
        self.top_trajs = None
        self.top_values = None
        self.top_idx = None
        self.effective_sample_size = float('nan')
        self.cost_breakdown = {}
        self.cov_diag = None
        self.phase_timing = {}
        self.control_space = control_space
        # Set by request_reset(), consumed by the next get_command() -> optimize_process round
        # trip (see request_reset's docstring for why this exists).
        self._reset_requested = False
        
        #
        self.result_queue = Queue(maxsize=1)
        self.opt_queue = Queue(maxsize=1)

        
        # Headless evaluation calls get_command_debug(), which optimizes synchronously
        # and never consumes this worker. Allow that mode to skip an otherwise unused
        # CUDA subprocess (forking after CUDA initialization is invalid in PyTorch).
        self.opt_process = None
        if os.environ.get("STORM_DISABLE_OPT_PROCESS", "0") != "1":
            self.opt_process = Process(
                target=optimize_process,
                args=(self._control_cache_path, self.opt_queue, self.result_queue),
            )
            self.opt_process.daemon = True
            self.opt_process.start()
        self.controller = controller
        self.control_dt = control_dt
        self.prev_mpc_tstep = 0.0
    def _predict_next_state_exact(self, t_step, curr_state):
        """Integrate precisely the requested latency over the action time grid.

        command_tstep holds interval endpoints. A partial first/last interval
        contributes only its overlap with [t_step, t_step + mpc_dt].
        """
        duration = max(0.0, float(self.mpc_dt))
        if duration == 0.0:
            return curr_state
        endpoints = np.asarray(self.command_tstep, dtype=float)
        end = float(t_step) + duration
        if endpoints.size == 0 or end > endpoints[-1] + 1e-9:
            return curr_state  # No known policy covers the requested interval.
        now = float(t_step)
        first = int(np.searchsorted(endpoints, now, side='right'))
        for index in range(first, len(endpoints)):
            stop = min(end, float(endpoints[index]))
            dt = stop - now
            if dt > 0.0:
                curr_state = self.controller.rollout_fn.dynamics_model.get_next_state(
                    curr_state, self.command[0][index], dt)
                now = stop
            if now >= end - 1e-9:
                break
        return curr_state

    def predict_next_state(self, t_step, curr_state):
        if getattr(self, 'exact_latency_prediction', False):
            return self._predict_next_state_exact(t_step, curr_state)
        # predict next state
        # given current t_step, integrate to t_step+mpc_dt
        t1 = find_first_idx(self.command_tstep, t_step)
        t2 = find_first_idx(self.command_tstep, t_step + self.mpc_dt)  # - 1
        if t1 < 0 or t2 < 0:
            # Requested time is beyond the planned command time-grid (or the grid is empty).
            # Skip prediction rather than crashing.
            return curr_state
        t1_idx = max(0, int(t1) - 1)
        t2_idx = int(t2)

        # integrate from t1->t2
        for i in range(t1_idx, t2_idx):
            command = self.command[0][i]
            curr_state = self.controller.rollout_fn.dynamics_model.get_next_state(curr_state, command, self.mpc_dt)
    
        return curr_state
    def get_command_debug(self, t_step, curr_state, debug=False, control_dt=0.01):
        """ This function runs the controller in the same process and waits for optimization to  complete before return of a new command
        Args:
        t_step: current timestep
        curr_state: current state to give to mpc
        debug: flag to enable debug commands [not implemented]
        control_dt: dt to integrate command to acceleration space from a higher order space(jerk, snap). 
        """
        if(self.command is not None):
            curr_state = self.predict_next_state(t_step, curr_state)

        current_state = np.append(curr_state, t_step + self.mpc_dt)
        shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt)
        
        state_tensor = torch.as_tensor(current_state,**self.controller.tensor_args).unsqueeze(0)


        mpc_time = time.time()
        command = list(self.controller.optimize(state_tensor, shift_steps=shift_steps))
        mpc_time = time.time() - mpc_time
        command[0] = command[0].cpu().numpy()
        self.command_tstep = self.traj_tstep + t_step
        
        self.opt_dt = mpc_time
        self.mpc_dt = t_step - self.prev_mpc_tstep
        self.prev_mpc_tstep = copy.deepcopy(t_step)
        
        # get command data:
        self.top_idx = self.controller.top_idx
        self.top_values = self.controller.top_values
        self.top_trajs = self.controller.top_trajs
        self.effective_sample_size = getattr(self.controller, 'effective_sample_size', float('nan'))
        self.command = command

        command_buffer, command_tstep_buffer = self.truncate_command(self.command[0], t_step, self.command_tstep)
        
        act = self.controller.rollout_fn.dynamics_model.integrate_action_step(command_buffer[0], self.control_dt)
        return act, command_tstep_buffer, self.command[1], command_buffer

    def request_reset(self):
        """Requests a full MPPI distribution reset (mean_action -> init_mean [zero action for
        control_space='acc'], cov_action -> init_cov) on the next get_command() round trip --
        see BaseTask.reset() for why/when this gets called (e.g. right after a disarmed
        gripper hold ends, so the resumed rollout doesn't warm-start from a stale, pre-hold
        plan). One-shot: consumed and cleared the next time get_command() enqueues a request.
        """
        self._reset_requested = True

    def get_command(self, t_step, curr_state, debug=False, control_dt=0.01):
        if(self.opt_queue.empty()):# and self.command is None):
            # integrate current state to mpc_dt:
            #
            if(self.command is not None):
                curr_state = self.predict_next_state(t_step, curr_state)

            curr_state = np.append(curr_state, t_step + self.mpc_dt)

            # planned command:

            shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt) #- 1
            if(shift_steps < 0):
                shift_steps = 0

            opt_data = {'state': curr_state, 't_step':t_step + self.mpc_dt, 'done':self.done, 'params':self.params, 'shift_steps':shift_steps, 'pred_mpc_dt':self.mpc_dt, 'reset': self._reset_requested}

            self.start_time = time.time()
            self.mpc_dt = t_step - self.prev_mpc_tstep

            self.opt_queue.put(opt_data)
            self.params = None
            self._reset_requested = False

        # wait for first command
        while(self.command is None and self.result_queue.empty()):
            time.sleep(0.01)

        
        if(not self.result_queue.empty()):# and self.command is None):
            command_data = self.result_queue.get()
            self.command_tstep = self.traj_tstep + command_data['t_step']

            self.command = command_data['command']
            self.opt_dt = command_data['mpc_dt']
            self.prev_mpc_tstep = copy.deepcopy(t_step)

            self.top_values = command_data['top_values']
            self.top_trajs = command_data['top_trajs']
            self.top_idx = command_data['top_idx']
            self.effective_sample_size = command_data.get('effective_sample_size', float('nan'))
            self.cost_breakdown = command_data.get('cost_breakdown', {})
            self.cov_diag = command_data.get('cov_diag')
            self.phase_timing = command_data.get('phase_timing', {})
            
            
            
        # send to process

        if(self.command is None):
            raise ValueError

        command_buffer, command_tstep_buffer = self.truncate_command(self.command[0], t_step, self.command_tstep)
        
        #print(command_buffer.shape)
        act = self.controller.rollout_fn.dynamics_model.integrate_action_step(command_buffer[0], self.control_dt)
        return act, command_tstep_buffer, self.command[1], command_buffer
    
    def truncate_command(self, command, trunc_tstep, command_tstep):
        #print(trunc_tstep, command_tstep[:4])
        f_idx = find_first_idx(command_tstep, trunc_tstep) #- 1
        if(f_idx == -1):
            f_idx = 0
        #print(f_idx)
        return command[f_idx:], command_tstep[f_idx:]

    def update_params(self, **kwargs):
        # Merge rather than replace: self.params is only actually applied (and cleared) on the
        # next get_command() -> optimize_process round trip. Independent update_params() calls
        # within the same tick (e.g. a goal update and a world-collision update) previously
        # clobbered each other here since the second call's kwargs fully replaced the first's.
        self.params = {**(self.params or {}), **kwargs}

 
    def close(self):
        self.done = True
        if self.opt_process is not None:
            opt_data = {'state': None, 'dt':None, 'done':self.done, 'params':None}
            self.opt_queue.put(opt_data)
            self.opt_process.join()
        try:
            if getattr(self, "_control_cache_path", None):
                os.remove(self._control_cache_path)
        except Exception:
            pass

def optimize_process(control_string, opt_queue, result_queue):
    """
    This runs mpc in a seperate process.

    Input:
    current_state: current state
    shift_steps: how much to shift from the previous optimization

    Returns:
    command: command trajectory


    """
    controller = torch.load(control_string)
    try:
        controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()
    except Exception:
        pass
    timing_path = os.environ.get('HUMAN_BENCHMARK_TIMINGS')
    timing_file = open(timing_path, 'w', buffering=1) if timing_path else None
    plan_path = os.environ.get('HUMAN_BENCHMARK_PLANS')
    plan_file = open(plan_path, 'wb') if plan_path else None
    raw_path = os.environ.get('HUMAN_BENCHMARK_RAW_PLANS')
    raw_file = open(raw_path, 'wb') if raw_path else None
    i = 0
    start_time = time.time()
    state_tensor = torch.zeros((1, 2 * controller.rollout_fn.dynamics_model.n_dofs), **controller.tensor_args)
    goal_count = 0
    while (True):
        opt_data = opt_queue.get()
        if(opt_data['done']):
            break
        current_state = opt_data['state']
        state_tensor = torch.as_tensor(current_state,**controller.tensor_args).unsqueeze(0)
        shift_steps = opt_data['shift_steps']

        # Full distribution reset requested (see ControlProcess.request_reset /
        # BaseTask.reset): mean_action -> init_mean (zero action for control_space='acc'),
        # cov_action -> init_cov. Before update_params/optimize so a simultaneous goal change
        # (e.g. the next pick/place target) still applies normally on top of the fresh state,
        # instead of hotstart carrying forward a plan built for whatever was happening right
        # before the gripper hold.
        if opt_data.get('reset'):
            controller.reset_distribution()

        # update goal pose if it's not none
        if(opt_data['params'] is not None):
            #print('updating goal...')
            controller.rollout_fn.update_params(**opt_data['params'])
            goal_count += 1
            if(goal_count == 100):
                #controller.reset_mean()
                controller.reset_covariance()

        benchmark_start = time.perf_counter()
        mpc_time = time.time()
        command = list(controller.optimize(state_tensor, shift_steps=shift_steps))
        mpc_time = time.time() - mpc_time

        # get command data:
        top_idx = controller.top_idx
        top_values = controller.top_values
        top_trajs = controller.top_trajs
        effective_sample_size = getattr(controller, 'effective_sample_size', float('nan'))

        command[0] = command[0].cpu().numpy()
        benchmark_end = time.perf_counter()
        if timing_file is not None:
            rollout = controller.rollout_fn
            if hasattr(rollout, 'geometry'):
                geometry_metadata = dict(collision_backend='common_capsules',
                    robot_primitives=len(rollout.geometry['robot']),
                    human_capsules=len(rollout.human),
                    human_pairs=len(rollout.geometry['robot'])*len(rollout.human),
                    self_pairs=len(rollout.geometry['self_pairs']))
            else:
                collision = rollout.primitive_collision_cost
                link_spheres = collision.robot_world_coll.robot_coll.get_batch_robot_link_spheres()
                world = collision._dynamic_world_spheres
                geometry_metadata = dict(robot_spheres_per_link=[int(v.shape[1]) for v in link_spheres],
                    human_spheres=0 if world is None else int(world.shape[0]))
            timing_file.write(json.dumps(dict(t_step=float(opt_data['t_step']),
                solve_ms=1000*(benchmark_end-benchmark_start),
                wall_start_s=benchmark_start,wall_end_s=benchmark_end,
                native_opt_ms=1000*mpc_time,
                effective_sample_size=float(effective_sample_size),
                cat_diagnostics=getattr(controller, '_cat_pc_debug', {}),
                solver_diagnostics=getattr(controller, 'solver_diagnostics', {}),
                **geometry_metadata))+'\n')

        if plan_file is not None:
            rollout = controller.rollout_fn
            # Append fixed-width rows; evaluation of the selected mean plan is offline.
            row = np.r_[float(opt_data['t_step']), np.asarray(current_state).reshape(-1)[:14],
                rollout.goal_ee_pos.detach().cpu().numpy().reshape(-1),
                rollout.goal_ee_rot.detach().cpu().numpy().reshape(-1), command[0].reshape(-1)]
            np.save(plan_file, row, allow_pickle=False)
            plan_file.flush()
        if raw_file is not None and getattr(controller, 'raw_plan', None) is not None:
            rp = controller.raw_plan
            np.save(raw_file, np.r_[float(opt_data['t_step']), benchmark_start, benchmark_end, rp['goal'].ravel(), rp['rotation'].ravel(), rp['xs'].ravel(), rp['us'].ravel()], allow_pickle=False)
            raw_file.flush()
        cost_breakdown = getattr(controller.rollout_fn, "last_cost_breakdown", {})

        # Diagnostic-only phase timing for this optimize() call -- merges the controller-level
        # breakdown (shift/sample_actions/rollout_fn/update_distribution, set in
        # control_base.py's optimize() and olgaussian_mpc.py's generate_rollouts) with the
        # rollout_fn-level sub-breakdown of its own "rollout_fn" slice (robot_model/cost_fns,
        # set in arm_base.py's rollout_fn). See storm_mppi_planner_node.py's "MPPI phase
        # timing" log for where this surfaces.
        phase_timing = dict(getattr(controller, "last_phase_timing", {}) or {})
        rollout_phase_timing = getattr(controller.rollout_fn, "last_rollout_phase_timing", {}) or {}
        for k, v in rollout_phase_timing.items():
            phase_timing[f"rollout_fn.{k}"] = v

        # Exploration covariance actually used for this optimize() call, for debug printing on
        # the bridge side (see BaseTask.cov_diagonal / storm_mppi_planner_node.py). full_cov
        # handles all cov_type variants uniformly: sigma_I/diag_AxA/full_AxA give one value per
        # action dim (same width applied at every horizon step); full_HAxHA genuinely varies
        # per horizon step, so that case is reshaped to [horizon, d_action] instead of flattened.
        try:
            cov_diag_flat = controller.full_cov.diagonal().detach().cpu().numpy()
            if controller.cov_type == 'full_HAxHA':
                cov_diag = cov_diag_flat.reshape(controller.horizon, controller.d_action).tolist()
            else:
                cov_diag = cov_diag_flat.reshape(-1).tolist()
        except Exception:
            cov_diag = None

        result = {'command':command, 't_step': opt_data['t_step'], 'mpc_dt': mpc_time,
                  'top_values':top_values, 'top_trajs':top_trajs, 'top_idx':top_idx,
                  'effective_sample_size': effective_sample_size,
                  'cost_breakdown': cost_breakdown,
                  'cov_diag': cov_diag,
                  'phase_timing': phase_timing}
        result_queue.put(result)
        i = time.time() - start_time
    return True
