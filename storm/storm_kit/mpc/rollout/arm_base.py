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
import time

import torch
import torch.autograd.profiler as profiler

from ..cost import DistCost, PoseCost, ProjectedDistCost, JacobianCost, ZeroCost, EEVelCost, StopCost, FiniteDifferenceCost
from ..cost.bound_cost import BoundCost
from ..cost.manipulability_cost import ManipulabilityCost
from ..cost import CollisionCost, VoxelCollisionCost, PrimitiveCollisionCost
from ..model import URDFKinematicModel
from ...util_file import join_path, get_assets_path
from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ...mpc.model.integration_utils import build_fd_matrix
from ...mpc.rollout.rollout_base import RolloutBase
from ..cost.robot_self_collision_cost import RobotSelfCollisionCost

# Placeholder world for box_collision_cost/table_collision_cost at construction time -- the
# real content arrives via update_params(box_world_params=..., table_world_params=...) right
# after the bridge finishes building the task (see storm_mppi_planner_node.py), same pattern
# already used for dynamic_world_spheres/goal_ee_pos.
_EMPTY_WORLD_PARAMS = {"world_model": {"coll_objs": {"sphere": {}, "cube": {}}}}


class ArmBase(RolloutBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update exp_params to be kwargs
    """

    def __init__(self, exp_params, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        mppi_params = exp_params['mppi']
        model_params = exp_params['model']
        cat_pc_cfg = self.exp_params.get('mppi', {}).get('cat_primitive_collision', {})
        self.cat_primitive_collision_enabled = bool(cat_pc_cfg.get('enabled', False))
        self.cat_primitive_collision_distance_threshold = cat_pc_cfg.get('distance_threshold', None)
        if self.cat_primitive_collision_distance_threshold is not None:
            self.cat_primitive_collision_distance_threshold = float(self.cat_primitive_collision_distance_threshold)
        self.cat_primitive_collision_violation_clip_max = cat_pc_cfg.get('violation_clip_max', None)
        if self.cat_primitive_collision_violation_clip_max is not None:
            self.cat_primitive_collision_violation_clip_max = float(self.cat_primitive_collision_violation_clip_max)
        self.cat_primitive_collision_violation_scale = cat_pc_cfg.get('violation_scale', None)
        if self.cat_primitive_collision_violation_scale is not None:
            self.cat_primitive_collision_violation_scale = float(self.cat_primitive_collision_violation_scale)
        cat_sb_cfg = self.exp_params.get('mppi', {}).get('cat_state_bound', {})
        self.cat_state_bound_enabled = bool(cat_sb_cfg.get('enabled', False))
        self.cat_self_enabled = bool(mppi_params.get('cat_self_collision', {}).get('enabled',False))
        self.explicit_self = bool(exp_params['cost']['robot_self_collision'].get('explicit_geometry',False))
        if self.cat_self_enabled and not self.explicit_self:
            raise ValueError('CaT self-collision requires explicit geometry')
        self.cat_state_bound_bound_thresh = cat_sb_cfg.get('bound_thresh', None)
        if self.cat_state_bound_bound_thresh is not None:
            self.cat_state_bound_bound_thresh = float(self.cat_state_bound_bound_thresh)

        robot_params = exp_params['robot_params']
        
        assets_path = get_assets_path()
        #print('EE LINK',exp_params['model']['ee_link_name'])
        # initialize dynamics model:
        dynamics_horizon = mppi_params['horizon'] * model_params['dt']
        #Create the dynamical system used for rollouts
        self.dynamics_model = URDFKinematicModel(join_path(assets_path,exp_params['model']['urdf_path']),
                                                 dt=exp_params['model']['dt'],
                                                 batch_size=mppi_params['num_particles'],
                                                 horizon=dynamics_horizon,
                                                 tensor_args=self.tensor_args,
                                                 ee_link_name=exp_params['model']['ee_link_name'],
                                                 link_names=exp_params['model']['link_names'],
                                                 dt_traj_params=exp_params['model']['dt_traj_params'],
                                                 control_space=exp_params['control_space'],
                                                 vel_scale=exp_params['model']['vel_scale'])
        torque_cfg = exp_params['cost'].get('joint_torque', {})
        self.torque_weight = float(torque_cfg.get('weight', 0.0))
        self.torque_limit_scale = float(torque_cfg.get('limit_scale', 1.0))
        self.cat_torque_enabled = bool(mppi_params.get('cat_joint_torque', {}).get('enabled', False))
        if not 0.0 < self.torque_limit_scale <= 1.0 or self.torque_weight < 0:
            raise ValueError('Torque limit_scale must be in (0,1] and weight nonnegative')
        self.torque_model = None
        if self.torque_weight > 0 or self.cat_torque_enabled:
            from ..model.batched_inverse_dynamics import BatchedInverseDynamics
            self.torque_model = BatchedInverseDynamics(
                join_path(assets_path, model_params['urdf_path']), self.tensor_args,
                backend=torque_cfg.get('backend', 'matrix'))
            self.torque_limits = self.torque_model.effort_limits * self.torque_limit_scale

        self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        # rollout traj_dt starts from dt->dt*(horizon+1) as tstep 0 is the current state
        #self.traj_dt = torch.arange(self.dt, (mppi_params['horizon'] + 1) * self.dt, self.dt, device=device, dtype=float_dtype)
        self.traj_dt = self.dynamics_model.traj_dt
        #print(self.traj_dt)
        
        self.fd_matrix = build_fd_matrix(10 - self.exp_params['cost']['smooth']['order'], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'], PREV_STATE=True, order=self.exp_params['cost']['smooth']['order'])
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        
        device = self.tensor_args['device']
        float_dtype = self.tensor_args['dtype']

        self.jacobian_cost = JacobianCost(ndofs=self.n_dofs, device=device,
                                          float_dtype=float_dtype,
                                          retract_weight=exp_params['cost']['retract_weight'])
        
        self.null_cost = ProjectedDistCost(ndofs=self.n_dofs, device=device, float_dtype=float_dtype,
                                           **exp_params['cost']['null_space'])

        self.manipulability_cost = ManipulabilityCost(ndofs=self.n_dofs, device=device,
                                                      float_dtype=float_dtype,
                                                      **exp_params['cost']['manipulability'])

        self.zero_vel_cost = ZeroCost(device=device, float_dtype=float_dtype, **exp_params['cost']['zero_vel'])

        self.zero_acc_cost = ZeroCost(device=device, float_dtype=float_dtype, **exp_params['cost']['zero_acc'])

        self.stop_cost = StopCost(**exp_params['cost']['stop_cost'],
                                  tensor_args=self.tensor_args,
                                  traj_dt=self.traj_dt)
        self.stop_cost_acc = StopCost(**exp_params['cost']['stop_cost_acc'],
                                      tensor_args=self.tensor_args,
                                      traj_dt=self.traj_dt)

        self.retract_state = torch.tensor([self.exp_params['cost']['retract_state']], device=device, dtype=float_dtype)

        # collision model:

        # build robot collision model

        

        
        if self.exp_params['cost']['smooth']['weight'] > 0:
            self.smooth_cost = FiniteDifferenceCost(**self.exp_params['cost']['smooth'],
                                                    tensor_args=self.tensor_args)

        if(self.exp_params['cost']['voxel_collision']['weight'] > 0):
            self.voxel_collision_cost = VoxelCollisionCost(robot_params=robot_params,
                                                           tensor_args=self.tensor_args,
                                                           **self.exp_params['cost']['voxel_collision'])

        self.primitive_collision_cost = None
        if(exp_params['cost']['primitive_collision']['weight'] > 0.0 or self.cat_primitive_collision_enabled):
            self.primitive_collision_cost = PrimitiveCollisionCost(world_params=world_params, robot_params=robot_params, tensor_args=self.tensor_args, **self.exp_params['cost']['primitive_collision'])

        # Dedicated collision costs for pick-object boxes and the table, kept separate from
        # primitive_collision_cost above so each gets its own threshold/weight (typically
        # distance_threshold: 0.0 for boxes -- only real contact/penetration counts, no
        # standoff margin, since the robot must approach and touch these). Always active, like
        # table_collision_cost below -- the *content* of box_world_params is what changes over
        # time (the bridge excludes whichever specific box is currently held by the gripper,
        # via update_params(box_world_params=...), while every other box stays a real obstacle).
        self.box_collision_cost = None
        if exp_params['cost'].get('box_collision', {}).get('weight', 0.0) > 0.0:
            self.box_collision_cost = PrimitiveCollisionCost(
                world_params=_EMPTY_WORLD_PARAMS, robot_params=robot_params,
                tensor_args=self.tensor_args, **self.exp_params['cost']['box_collision'],
            )

        self.table_collision_cost = None
        if exp_params['cost'].get('table_collision', {}).get('weight', 0.0) > 0.0:
            self.table_collision_cost = PrimitiveCollisionCost(
                world_params=_EMPTY_WORLD_PARAMS, robot_params=robot_params,
                tensor_args=self.tensor_args, **self.exp_params['cost']['table_collision'],
            )

        if exp_params['cost']['robot_self_collision']['weight'] > 0.0 or self.cat_self_enabled:
            if self.explicit_self:
                from ..cost.explicit_self_collision_cost import ExplicitSelfCollisionCost
                self.robot_self_collision_cost = ExplicitSelfCollisionCost(robot_params=robot_params, tensor_args=self.tensor_args, **exp_params['cost']['robot_self_collision'])
            else:
                self.robot_self_collision_cost = RobotSelfCollisionCost(robot_params=robot_params, tensor_args=self.tensor_args, **exp_params['cost']['robot_self_collision'])


        self.ee_vel_cost = EEVelCost(ndofs=self.n_dofs,device=device, float_dtype=float_dtype,**exp_params['cost']['ee_vel'])

        bounds = torch.cat([self.dynamics_model.state_lower_bounds[:self.n_dofs * 3].unsqueeze(0),self.dynamics_model.state_upper_bounds[:self.n_dofs * 3].unsqueeze(0)], dim=0).T
        self.bound_cost = BoundCost(**exp_params['cost']['state_bound'],
                                    tensor_args=self.tensor_args,
                                    bounds=bounds)

        self.link_pos_seq = torch.zeros((1, 1, len(self.dynamics_model.link_names), 3), **self.tensor_args)
        self.link_rot_seq = torch.zeros((1, 1, len(self.dynamics_model.link_names), 3, 3), **self.tensor_args)

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_breakdown=False):

        ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
        state_batch = state_dict['state_seq']
        lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        prev_state = state_dict['prev_state_seq']
        prev_state_tstep = state_dict['prev_state_seq'][:,-1]

        retract_state = self.retract_state
        breakdown = {} if return_breakdown else None

        J_full = torch.cat((lin_jac_batch, ang_jac_batch), dim=-2)
        

        #null-space cost
        #if self.exp_params['cost']['null_space']['weight'] > 0:
        null_disp_cost = self.null_cost.forward(state_batch[:,:,0:self.n_dofs] -
                                                retract_state[:,0:self.n_dofs],
                                                J_full,
                                                proj_type='identity',
                                                dist_type='squared_l2')
        if breakdown is not None:
            # Snapshot before `cost = null_disp_cost` aliases the tensor -- every later
            # `cost += ...` in this function is an in-place op that would otherwise silently
            # accumulate into null_disp_cost too, corrupting this breakdown entry.
            breakdown["null_space"] = null_disp_cost.clone()
        cost = null_disp_cost

        self_cost = torch.zeros_like(cost)
        self_violation = torch.zeros_like(cost)
        primitive_cost = None
        primitive_violation = None
        state_bound_cost = None
        state_bound_violation = None

        if(no_coll == True and horizon_cost == False):
            if return_breakdown:
                primitive_cost = torch.zeros_like(cost)
                primitive_violation = torch.zeros_like(cost)
                state_bound_cost = torch.zeros_like(cost)
                state_bound_violation = torch.zeros_like(cost)
                return cost, {
                    "self_collision_costs": torch.zeros_like(cost),
                    "self_collision_violation": torch.zeros_like(cost),
                    "joint_torque_costs": torch.zeros_like(cost),
                    "joint_torque_violation": torch.zeros_like(cost),
                    "primitive_collision_costs": primitive_cost,
                    "primitive_collision_violation": primitive_violation,
                    "state_bound_costs": state_bound_cost,
                    "state_bound_violation": state_bound_violation,
                }
            return cost
        if(self.exp_params['cost']['manipulability']['weight'] > 0.0):
            _c = self.manipulability_cost.forward(J_full)
            cost += _c
            if breakdown is not None: breakdown["manipulability"] = _c

        if(horizon_cost):
            if self.exp_params['cost']['stop_cost']['weight'] > 0:
                _c = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
                cost += _c
                if breakdown is not None: breakdown["stop_cost"] = _c

            if self.exp_params['cost']['stop_cost_acc']['weight'] > 0:
                _c = self.stop_cost_acc.forward(state_batch[:, :, self.n_dofs*2:self.n_dofs * 3])
                cost += _c
                if breakdown is not None: breakdown["stop_cost_acc"] = _c

            if self.exp_params['cost']['smooth']['weight'] > 0:
                order = self.exp_params['cost']['smooth']['order']
                prev_dt = (self.fd_matrix @ prev_state_tstep)[-order:]
                n_mul = 1
                state = state_batch[:,:, self.n_dofs * n_mul:self.n_dofs * (n_mul+1)]
                p_state = prev_state[-order:,self.n_dofs * n_mul: self.n_dofs * (n_mul+1)].unsqueeze(0)
                p_state = p_state.expand(state.shape[0], -1, -1)
                state_buffer = torch.cat((p_state, state), dim=1)
                traj_dt = torch.cat((prev_dt, self.traj_dt))
                _c = self.smooth_cost.forward(state_buffer, traj_dt)
                cost += _c
                if breakdown is not None: breakdown["smooth"] = _c


        if return_breakdown:
            state_bound_cost, state_bound_violation = self.bound_cost.forward(
                state_batch[:, :, : self.n_dofs * 3],
                return_violation=True,
                violation_bound_thresh=(
                    self.cat_state_bound_bound_thresh if self.cat_state_bound_enabled else None
                ),
            )
            if self.exp_params['cost']['state_bound']['weight'] > 0:
                cost += state_bound_cost
        elif self.exp_params['cost']['state_bound']['weight'] > 0:
            cost += self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3])

        if self.exp_params['cost']['ee_vel']['weight'] > 0:
            _c = self.ee_vel_cost.forward(state_batch, lin_jac_batch)
            cost += _c
            if breakdown is not None: breakdown["ee_vel"] = _c



        if(not no_coll):
            if self.exp_params['cost']['robot_self_collision']['weight'] > 0 or self.cat_self_enabled:
                kwargs=dict(link_pos_seq=link_pos_batch,link_rot_seq=link_rot_batch)
                if self.explicit_self:
                    self_cost,self_violation=self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs],return_violation=True,**kwargs)
                else:
                    self_cost=self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs],**kwargs)
                cost += self_cost
                if breakdown is not None:breakdown['robot_self_collision']=self_cost
            primitive_weight_enabled = self.exp_params['cost']['primitive_collision']['weight'] > 0.0
            if self.primitive_collision_cost is not None and (primitive_weight_enabled or return_breakdown):
                if return_breakdown:
                    coll_cost, coll_violation = self.primitive_collision_cost.forward(
                        link_pos_batch,
                        link_rot_batch,
                        return_violation=True,
                        violation_distance_threshold=(
                            self.cat_primitive_collision_distance_threshold
                            if self.cat_primitive_collision_enabled
                            else None
                        ),
                        violation_clip_max=(
                            self.cat_primitive_collision_violation_clip_max
                            if self.cat_primitive_collision_enabled
                            else None
                        ),
                        violation_scale=(
                            self.cat_primitive_collision_violation_scale
                            if self.cat_primitive_collision_enabled
                            else None
                        ),
                    )
                    primitive_cost = coll_cost
                    primitive_violation = coll_violation
                    if primitive_weight_enabled:
                        cost += coll_cost
                elif primitive_weight_enabled:
                    coll_cost = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
                    cost += coll_cost

            if self.box_collision_cost is not None:
                box_cost = self.box_collision_cost.forward(link_pos_batch, link_rot_batch)
                cost += box_cost
                if breakdown is not None:
                    breakdown["box_collision"] = box_cost

            if self.table_collision_cost is not None:
                table_cost = self.table_collision_cost.forward(link_pos_batch, link_rot_batch)
                cost += table_cost
                if breakdown is not None:
                    breakdown["table_collision"] = table_cost

            if self.exp_params['cost']['voxel_collision']['weight'] > 0:
                coll_cost = self.voxel_collision_cost.forward(link_pos_batch, link_rot_batch)
                cost += coll_cost

        torque_cost = torch.zeros_like(cost)
        torque_violation = torch.zeros_like(cost)
        if self.torque_model is not None:
            tau = self.torque_model(state_batch[..., :self.n_dofs],
                                    state_batch[..., self.n_dofs:2*self.n_dofs],
                                    state_batch[..., 2*self.n_dofs:3*self.n_dofs])
            excess = torch.relu(torch.abs(tau) / self.torque_limits - 1.0)
            torque_violation = excess.amax(dim=-1)
            torque_cost = self.torque_weight * excess.square().sum(dim=-1)
            cost += torque_cost
        if return_breakdown:
            breakdown["self_collision_costs"] = self_cost
            breakdown["self_collision_violation"] = self_violation
            breakdown["joint_torque"] = torque_cost
            breakdown["joint_torque_costs"] = torque_cost
            breakdown["joint_torque_violation"] = torque_violation
            if primitive_cost is None:
                primitive_cost = torch.zeros_like(cost)
            if primitive_violation is None:
                primitive_violation = torch.zeros_like(cost)
            if state_bound_cost is None:
                state_bound_cost = torch.zeros_like(cost)
            if state_bound_violation is None:
                state_bound_violation = torch.zeros_like(cost)
            breakdown["primitive_collision"] = primitive_cost
            breakdown["state_bound"] = state_bound_cost
            breakdown["primitive_collision_costs"] = primitive_cost
            breakdown["primitive_collision_violation"] = primitive_violation
            breakdown["state_bound_costs"] = state_bound_cost
            breakdown["state_bound_violation"] = state_bound_violation
            return cost, breakdown

        return cost
    
    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """
        # Diagnostic-only sub-phase timing -- see the matching note in control_base.py's
        # optimize(). Stored on self (the rollout_fn owner, a *different* object than the
        # controller that owns last_phase_timing in olgaussian_mpc.py's generate_rollouts) so
        # the bridge can merge both dicts -- see mpc_process_wrapper.py.
        _sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        self.last_rollout_phase_timing = {}

        with profiler.record_function("robot_model"):
            _t0 = time.time()
            state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
            _sync()
            self.last_rollout_phase_timing['robot_model'] = time.time() - _t0

        #link_pos_seq, link_rot_seq = self.dynamics_model.get_link_poses()
        with profiler.record_function("cost_fns"):
            _t0 = time.time()
            cost_seq, cost_breakdown = self.cost_fn(state_dict, act_seq, return_breakdown=True)
            _sync()
            self.last_rollout_phase_timing['cost_fns'] = time.time() - _t0

        # Store mean cost per named term (sum over horizon, mean over particles) as plain floats.
        # Keys match YAML cost names so the bridge can publish them by name.
        _skip = {"primitive_collision_costs", "primitive_collision_violation",
                 "state_bound_costs", "state_bound_violation",
                 "joint_torque_costs", "joint_torque_violation",
                 "self_collision_costs", "self_collision_violation"}
        self.last_cost_breakdown = {
            k: float(v.sum(dim=-1).mean().item())
            for k, v in cost_breakdown.items()
            if k not in _skip
        }

        sim_trajs = dict(
            self_collision_costs=cost_breakdown["self_collision_costs"],
            self_collision_violation=cost_breakdown["self_collision_violation"],
            joint_torque_costs=cost_breakdown["joint_torque_costs"],
            joint_torque_violation=cost_breakdown["joint_torque_violation"],
            actions=act_seq,
            costs=cost_seq,
            ee_pos_seq=state_dict['ee_pos_seq'],
            rollout_time=0.0,
            primitive_collision_costs=cost_breakdown["primitive_collision_costs"],
            primitive_collision_violation=cost_breakdown["primitive_collision_violation"],
            state_bound_costs=cost_breakdown["state_bound_costs"],
            state_bound_violation=cost_breakdown["state_bound_violation"],
        )

        return sim_trajs

    def update_params(self, retract_state=None, world_params=None, dynamic_world_spheres=None,
                       box_world_params=None, table_world_params=None):
        """
        Updates the goal targets for the cost functions.

        """

        if(retract_state is not None):
            self.retract_state = torch.as_tensor(retract_state, **self.tensor_args).unsqueeze(0)

        if world_params is not None and self.primitive_collision_cost is not None:
            self.primitive_collision_cost = PrimitiveCollisionCost(
                world_params=world_params,
                robot_params=self.exp_params['robot_params'],
                tensor_args=self.tensor_args,
                **self.exp_params['cost']['primitive_collision'],
            )

        if box_world_params is not None and self.box_collision_cost is not None:
            self.box_collision_cost = PrimitiveCollisionCost(
                world_params=box_world_params,
                robot_params=self.exp_params['robot_params'],
                tensor_args=self.tensor_args,
                **self.exp_params['cost']['box_collision'],
            )

        if table_world_params is not None and self.table_collision_cost is not None:
            self.table_collision_cost = PrimitiveCollisionCost(
                world_params=table_world_params,
                robot_params=self.exp_params['robot_params'],
                tensor_args=self.tensor_args,
                **self.exp_params['cost']['table_collision'],
            )

        if dynamic_world_spheres is not None and self.primitive_collision_cost is not None:
            self.primitive_collision_cost.set_dynamic_world_spheres(dynamic_world_spheres)

        return True
    
    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
    
    def get_ee_pose(self, current_state):
        current_state = current_state.to(**self.tensor_args)
         
        
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.exp_params['model']['ee_link_name'])

        ee_quat = matrix_to_quaternion(ee_rot_batch)
        state = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                 'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                 'ee_quat_seq':ee_quat}
        return state
    def current_cost(self, current_state, no_coll=True):
        current_state = current_state.to(**self.tensor_args)
        
        curr_batch_size = 1
        num_traj_points = 1 #self.dynamics_model.num_traj_points
        
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.exp_params['model']['ee_link_name'])


        link_pos_seq = self.link_pos_seq
        
        link_rot_seq = self.link_rot_seq

        # get link poses:
        for ki,k in enumerate(self.dynamics_model.link_names):
            link_pos, link_rot = self.dynamics_model.robot_model.get_link_pose(k)
            link_pos_seq[:,:,ki,:] = link_pos.view((curr_batch_size, num_traj_points,3))
            link_rot_seq[:,:,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points,3,3))
            
        if(len(current_state.shape) == 2):
            current_state = current_state.unsqueeze(0)
            ee_pos_batch = ee_pos_batch.unsqueeze(0)
            ee_rot_batch = ee_rot_batch.unsqueeze(0)
            lin_jac_batch = lin_jac_batch.unsqueeze(0)
            ang_jac_batch = ang_jac_batch.unsqueeze(0)

        state_dict = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                      'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                      'state_seq': current_state,'link_pos_seq':link_pos_seq,
                      'link_rot_seq':link_rot_seq,
                      'prev_state_seq':current_state}
        
        cost = self.cost_fn(state_dict, None,no_coll=no_coll, horizon_cost=False, return_dist=True)

        return cost, state_dict
