#!/usr/bin/env python
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

import copy

import numpy as np
import scipy.special
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import normalize as f_norm

from .control_utils import cost_to_go, matrix_cholesky, batch_cholesky
from .olgaussian_mpc import OLGaussianMPC

class MPPI(OLGaussianMPC):
    """
    .. inheritance-diagram:: MPPI
       :parts: 1

    Class that implements Model Predictive Path Integral Controller
    
    Implementation is based on 
    Williams et. al, Information Theoretic MPC for Model-Based Reinforcement Learning
    with additional functions for updating the covariance matrix
    and calculating the soft-value function.

    """

    def __init__(self,
                 d_action,
                 horizon,
                 init_cov,
                 init_mean,
                 base_action,
                 beta,
                 num_particles,
                 step_size_mean,
                 step_size_cov,
                 alpha,
                 gamma,
                 kappa,
                 n_iters,
                 action_lows,
                 action_highs,
                 null_act_frac=0.,
                 rollout_fn=None,
                 sample_mode='mean',
                 hotstart=True,
                 squash_fn='clamp',
                 update_cov=False,
                 cov_type='sigma_I',
                 seed=0,
                 sample_params={'type': 'halton', 'fixed_samples': True, 'seed':0, 'filter_coeffs':None},
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 cat_primitive_collision=None,
                 visual_traj='state_seq',
                 cat_state_bound=None,
                 cat_joint_torque=None,
                 cat_self_collision=None,
                 cat_equation_mode="legacy",
                 cov_min=None,
                 cov_max=None):
        
        super(MPPI, self).__init__(d_action,
                                   action_lows, 
                                   action_highs,
                                   horizon,
                                   init_cov,
                                   init_mean,
                                   base_action,
                                   num_particles,
                                   gamma,
                                   n_iters,
                                   step_size_mean,
                                   step_size_cov, 
                                   null_act_frac,
                                   rollout_fn,
                                   sample_mode,
                                   hotstart,
                                   squash_fn,
                                   cov_type,
                                   seed,
                                   sample_params=sample_params,
                                   tensor_args=tensor_args)
        if cat_equation_mode not in ("legacy", "paper"):
            raise ValueError("cat_equation_mode must be legacy or paper")
        self._cat_equation_mode = cat_equation_mode
        self.beta = beta
        self.alpha = alpha  # 0 means control cost is on, 1 means off
        self.update_cov = update_cov
        self.kappa = kappa
        self.visual_traj = visual_traj
        # Hard bounds on the adaptive covariance's diagonal (per-action-dim variance), applied
        # every _shift() after the kappa floor-add. kappa alone only sets a soft resting floor
        # (see the step_size_cov/kappa fixed-point derivation) -- there is otherwise no ceiling
        # anywhere in this update rule, so cov_action can in principle grow unboundedly when
        # particles persistently disagree. None (default) preserves the old unbounded behavior
        # exactly -- these are opt-in.
        self.cov_min = float(cov_min) if cov_min is not None else None
        self.cov_max = float(cov_max) if cov_max is not None else None

        cat_pc_cfg = cat_primitive_collision if isinstance(cat_primitive_collision, dict) else {}
        self._cat_pc_enabled = bool(cat_pc_cfg.get('enabled', False))
        self._cat_pc_p_max = float(cat_pc_cfg.get('p_max', 0.15))
        self._cat_pc_tau_c = float(cat_pc_cfg.get('tau_c', 0.95))
        self._cat_pc_tau_b = float(cat_pc_cfg.get('tau_b', 0.95))
        self._cat_pc_eps = float(cat_pc_cfg.get('eps', 1.0e-6))

        cat_sb_cfg = cat_state_bound if isinstance(cat_state_bound, dict) else {}
        self._cat_sb_enabled = bool(cat_sb_cfg.get('enabled', False))
        self._cat_sb_p_max = float(cat_sb_cfg.get('p_max', 0.15))
        self._cat_sb_tau_c = float(cat_sb_cfg.get('tau_c', 0.95))
        self._cat_sb_eps = float(cat_sb_cfg.get('eps', 1.0e-6))

        if self._cat_pc_p_max <= 0.0 or self._cat_pc_p_max > 1.0:
            raise ValueError(f"cat_primitive_collision.p_max must be in (0, 1], got {self._cat_pc_p_max}.")
        if self._cat_pc_tau_c < 0.0 or self._cat_pc_tau_c >= 1.0:
            raise ValueError(f"cat_primitive_collision.tau_c must be in [0, 1), got {self._cat_pc_tau_c}.")
        if self._cat_pc_tau_b < 0.0 or self._cat_pc_tau_b >= 1.0:
            raise ValueError(f"cat_primitive_collision.tau_b must be in [0, 1), got {self._cat_pc_tau_b}.")
        if self._cat_pc_eps <= 0.0:
            raise ValueError(f"cat_primitive_collision.eps must be > 0, got {self._cat_pc_eps}.")
        if self._cat_sb_p_max <= 0.0 or self._cat_sb_p_max > 1.0:
            raise ValueError(f"cat_state_bound.p_max must be in (0, 1], got {self._cat_sb_p_max}.")
        if self._cat_sb_tau_c < 0.0 or self._cat_sb_tau_c >= 1.0:
            raise ValueError(f"cat_state_bound.tau_c must be in [0, 1), got {self._cat_sb_tau_c}.")
        if self._cat_sb_eps <= 0.0:
            raise ValueError(f"cat_state_bound.eps must be > 0, got {self._cat_sb_eps}.")

        torque_cfg = cat_joint_torque if isinstance(cat_joint_torque, dict) else {}
        self._cat_torque_enabled = bool(torque_cfg.get('enabled', False))
        self._cat_torque_p_max = float(torque_cfg.get('p_max', 1.0))
        self._cat_torque_tau_c = float(torque_cfg.get('tau_c', 0.95))
        self._cat_torque_eps = float(torque_cfg.get('eps', 1.e-6))
        if not 0 < self._cat_torque_p_max <= 1 or not 0 <= self._cat_torque_tau_c < 1 or self._cat_torque_eps <= 0:
            raise ValueError('Invalid CaT joint-torque hazard parameters')
        self_cfg = cat_self_collision if isinstance(cat_self_collision,dict) else {}
        self._cat_self_enabled=bool(self_cfg.get('enabled',False))
        self._cat_self_p_max=float(self_cfg.get('p_max',1.))
        self._cat_self_tau_c=float(self_cfg.get('tau_c',.95))
        if not 0 < self._cat_self_p_max <= 1 or not 0 <= self._cat_self_tau_c < 1:
            raise ValueError('Invalid CaT self collision hazard')
        self._cat_self_cmax_ema=None
        self._cat_torque_cmax_ema = None
        self._cat_pc_cmax_ema = None
        self._cat_pc_bmax_ema = None
        self._cat_sb_cmax_ema = None
        self._cat_pc_debug = {}

    def _update_distribution(self, trajectories):
        """
           Update moments in the direction using sampled
           trajectories


        """
        costs = trajectories["costs"].to(**self.tensor_args)
        vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        w = self._exp_util(costs, actions, trajectories=trajectories)

        # Tier 2c diagnostic: effective sample size N_eff = 1/sum(w_k^2) of the importance
        # weights (w is a softmax over the particle dim, see _exp_util, so this is the standard
        # ESS formula). Ranges from 1 (collapsed onto one particle) to num_particles (uniform).
        # Low N_eff means the executed "mean" action is effectively an average of very few
        # particles each tick -- a direct read on how much sampling-noise risk step_size_mean/
        # smooth-cost retuning is fighting.
        self.effective_sample_size = float(1.0 / torch.sum(w * w).item())

        #Update best action
        best_idx = torch.argmax(w)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)

        # FIX: total_costs is lower-is-better (see softmax(-cost/beta) above), but torch.topk
        # defaults to largest=True -- so this previously selected the 10 WORST rollouts despite
        # being named top_trajs/top_values. Confirmed via grep across the whole workspace that
        # nothing consumes these for control decisions (visualization-only), so this is safe.
        top_values, top_idx = torch.topk(self.total_costs, 10, largest=False)
        #print(ee_pos_seq.shape, top_idx)
        self.top_values = top_values
        self.top_idx = top_idx
        self.top_trajs = torch.index_select(vis_seq, 0, top_idx).squeeze(0)
        #print(self.top_traj.shape)
        #print(self.best_traj.shape, best_idx, w.shape)
        #self.best_trajs = torch.index_select(

        weighted_seq = w.T * actions.T

        sum_seq = torch.sum(weighted_seq.T, dim=0)

        new_mean = sum_seq
        #print(self.stomp_matrix.shape, self.full_scale_tril.shape)
        #cov = self.stomp_matrix #torch.matmul(self.full_scale_tril, self.stomp_matrix)
        #m_matrix = (1.0 / self.horizon) * cov # f_norm(cov,dim=0)
        #sum_seq = sum_seq.transpose(0,1)
        
        #new_mean = torch.matmul(m_matrix,sum_seq.reshape(self.horizon * self.d_action,1)).view(self.d_action, self.horizon).transpose(0,1)
        
        
        # plot mean:
        # = new_mean.cpu().numpy()
        #b = sum_seq.cpu().numpy()#.T
        #print(w, top_idx)
        #new_mean = sum_seq.T
        #matplotlib.use('tkagg')
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean
        #c = self.mean_action.cpu().numpy()
        #plt.plot(a[:,0])
        #plt.plot(b[:,0])
        #plt.plot(actions[top_idx[0],:,0].cpu().numpy())
        #plt.show()

        delta = actions - self.mean_action.unsqueeze(0)

        #Update Covariance
        if self.update_cov:
            if self.cov_type == 'sigma_I':
                #weighted_delta = w * (delta ** 2).T
                #cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0))
                #print(cov_update.shape, self.cov_action)
                raise NotImplementedError('Need to implement covariance update of form sigma*I')
            
            elif self.cov_type == 'diag_AxA':
                #Diagonal covariance of size AxA
                weighted_delta = w * (delta ** 2).T
                # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
                cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
            elif self.cov_type == 'diag_HxH':
                raise NotImplementedError
            elif self.cov_type == 'full_AxA':
                #Full Covariance of size AxA
                weighted_delta = torch.sqrt(w) * (delta).T
                weighted_delta = weighted_delta.T.reshape((self.horizon * self.num_particles, self.d_action))
                cov_update = torch.matmul(weighted_delta.T, weighted_delta) / self.horizon
            elif self.cov_type == 'full_HAxHA':# and self.sample_type != 'stomp':
                weighted_delta = torch.sqrt(w) * delta.view(delta.shape[0], delta.shape[1] * delta.shape[2]).T #.unsqueeze(-1)
                cov_update = torch.matmul(weighted_delta, weighted_delta.T)
                
                # weighted_cov = w * (torch.matmul(delta_new, delta_new.transpose(-2,-1))).T
                # weighted_cov = w * cov.T
                # cov_update = torch.sum(weighted_cov.T,dim=0)
                #
            #elif self.sample_type == 'stomp':
            #    weighted_delta = w * (delta ** 2).T
            #    cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
            #    self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
            #        self.step_size_cov * cov_update
            #    #self.scale_tril = torch.sqrt(self.cov_action)
            #    return
            else:
                raise ValueError('Unidentified covariance type in update_distribution')
            
            self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
                self.step_size_cov * cov_update
            #if(cov_update == 'diag_AxA'):
            #    self.scale_tril = torch.sqrt(self.cov_action)
            # self.scale_tril = torch.cholesky(self.cov_action)

        
    def _clamp_cov_diagonal(self, is_matrix: bool):
        """Applies self.cov_min/cov_max (if set) to cov_action's diagonal in place. For the
        vector cov_types (sigma_I/diag_AxA) cov_action IS the diagonal; for the matrix types
        (full_AxA/full_HAxHA) only the diagonal is clamped -- off-diagonal correlation terms
        are left as computed. torch.Tensor.diagonal() returns a view for 2D tensors, so
        clamp_() on it mutates cov_action's diagonal directly, no copy-back needed.
        """
        if self.cov_min is None and self.cov_max is None:
            return
        target = self.cov_action.diagonal() if is_matrix else self.cov_action
        target.clamp_(min=self.cov_min, max=self.cov_max)

    def _shift(self, shift_steps):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        if(shift_steps == 0):
            return
        super()._shift(shift_steps)

        if self.update_cov:
            if self.cov_type == 'sigma_I':
                self.cov_action += self.kappa #* self.init_cov_action
                self._clamp_cov_diagonal(is_matrix=False)
                self.scale_tril = torch.sqrt(self.cov_action)
                # self.inv_cov_action = 1.0 / self.cov_action

            elif self.cov_type == 'diag_AxA':
                self.cov_action += self.kappa #* self.init_cov_action
                #self.cov_action[self.cov_action < 0.0005] = 0.0005
                self._clamp_cov_diagonal(is_matrix=False)
                self.scale_tril = torch.sqrt(self.cov_action)
                # self.inv_cov_action = 1.0 / self.cov_action

            elif self.cov_type == 'full_AxA':
                self.cov_action += self.kappa*self.I
                self._clamp_cov_diagonal(is_matrix=True)
                self.scale_tril = matrix_cholesky(self.cov_action) # torch.cholesky(self.cov_action) #
                # self.scale_tril = torch.cholesky(self.cov_action)
                # self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)

            elif self.cov_type == 'full_HAxHA':
                self.cov_action += self.kappa * self.I

                #shift covariance up and to the left
                # self.cov_action = torch.roll(self.cov_action, shifts=(-self.d_action, -self.d_action), dims=(0,1))
                # self.cov_action = torch.roll(self.cov_action, shifts=(-self.d_action, -self.d_action), dims=(0,1))
                # #set bottom A rows and right A columns to zeros
                # self.cov_action[-self.d_action:,:].zero_()
                # self.cov_action[:,-self.d_action:].zero_()
                # #set bottom right AxA block to init_cov value
                # self.cov_action[-self.d_action:, -self.d_action:] = self.init_cov*self.I2 

                shift_dim = shift_steps * self.d_action
                I2 = torch.eye(shift_dim, **self.tensor_args)
                self.cov_action = torch.roll(self.cov_action, shifts=(-shift_dim, -shift_dim), dims=(0,1))
                #set bottom A rows and right A columns to zeros
                self.cov_action[-shift_dim:,:].zero_()
                self.cov_action[:,-shift_dim:].zero_()
                #set bottom right AxA block to init_cov value
                self.cov_action[-shift_dim:, -shift_dim:] = self.init_cov*I2
                self._clamp_cov_diagonal(is_matrix=True)
                #update cholesky decomp
                self.scale_tril = torch.cholesky(self.cov_action)
                # self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)


    def _cat_build_delta(self, violation, *, ema_attr, tau_c, p_max, eps):
        eps_t = torch.as_tensor(eps, **self.tensor_args)
        batch_vmax = torch.clamp(torch.max(violation), min=eps_t)
        cmax_ema = getattr(self, ema_attr)
        if cmax_ema is None:
            cmax_ema = batch_vmax.detach().clone()
        else:
            cmax_ema = tau_c * cmax_ema + (1.0 - tau_c) * batch_vmax.detach()
        setattr(self, ema_attr, cmax_ema)

        # COSMIK-MPPI Eq. 17 and CaT Eq. 7 normalize by the EMA itself.
        # A batch-max lower bound weakens abrupt violations; retain it only
        # for explicit reproduction of the previous implementation.
        cmax = torch.maximum(cmax_ema, eps_t)
        if getattr(self, '_cat_equation_mode', 'legacy') == 'legacy':
            cmax = torch.maximum(cmax, batch_vmax)
        delta = p_max * torch.clamp(violation / cmax, min=0.0, max=1.0)
        delta = torch.clamp(delta, min=0.0, max=p_max)
        return delta

    def _cat_reward_baseline(self, batch_bmax):
        eps = torch.as_tensor(self._cat_pc_eps, **self.tensor_args)
        previous = self._cat_pc_bmax_ema
        average = batch_bmax.detach().clone() if previous is None else (
            self._cat_pc_tau_b * previous + (1.0-self._cat_pc_tau_b)*batch_bmax.detach())
        baseline = torch.maximum(average, batch_bmax) + eps
        # Eq. 20 stores the upper envelope, not the unbounded EMA candidate.
        self._cat_pc_bmax_ema = (baseline if self._cat_equation_mode == 'paper' else average).detach().clone()
        return baseline

    def _exp_util(self, costs, actions, trajectories=None):
        """
            Calculate weights using exponential utility
        """
        any_cat_enabled = self._cat_pc_enabled or self._cat_sb_enabled or self._cat_torque_enabled or self._cat_self_enabled
        if not any_cat_enabled:
            self._cat_pc_debug = {}
            traj_costs = cost_to_go(costs, self.gamma_seq)
            traj_costs = traj_costs[:,0]
        else:
            if trajectories is None:
                raise RuntimeError(
                    "Shifted-CaT requires trajectories dict in MPPI._exp_util when any cat_* is enabled."
                )

            required_keys = ["costs"]
            if self._cat_pc_enabled:
                required_keys.extend(["primitive_collision_costs", "primitive_collision_violation"])
            if self._cat_sb_enabled:
                required_keys.extend(["state_bound_costs", "state_bound_violation"])

            if self._cat_torque_enabled:
                required_keys.extend(["joint_torque_costs", "joint_torque_violation"])
            if self._cat_self_enabled:
                required_keys.extend(["self_collision_costs", "self_collision_violation"])
            missing = [k for k in required_keys if k not in trajectories]
            if missing:
                raise RuntimeError(
                    "Shifted-CaT enabled but rollout outputs are missing required keys: "
                    f"{missing}."
                )

            all_costs = trajectories["costs"].to(**self.tensor_args)
            if all_costs.ndim != 2:
                raise RuntimeError(
                    f"Shifted-CaT expects [num_particles, horizon] tensors, got shape {tuple(all_costs.shape)}."
                )

            task_costs = all_costs.clone()
            delta_list = []
            debug_stats = {}
            debug_stats["cat_all_cost_max"] = float(torch.max(all_costs).item())

            if self._cat_pc_enabled:
                primitive_costs = trajectories["primitive_collision_costs"].to(**self.tensor_args)
                primitive_violation = trajectories["primitive_collision_violation"].to(**self.tensor_args)
                if all_costs.shape != primitive_costs.shape or all_costs.shape != primitive_violation.shape:
                    raise RuntimeError(
                        "Shape mismatch for shifted-CaT primitive collision inputs: "
                        f"costs={tuple(all_costs.shape)}, "
                        f"primitive_collision_costs={tuple(primitive_costs.shape)}, "
                        f"primitive_collision_violation={tuple(primitive_violation.shape)}."
                    )
                task_costs = task_costs - primitive_costs
                delta_pc = self._cat_build_delta(
                    primitive_violation,
                    ema_attr="_cat_pc_cmax_ema",
                    tau_c=self._cat_pc_tau_c,
                    p_max=self._cat_pc_p_max,
                    eps=self._cat_pc_eps,
                )
                delta_list.append(delta_pc)
                debug_stats["cat_pc_delta_max"] = float(torch.max(delta_pc).item())
                debug_stats["cat_pc_cmax_ema"] = float(self._cat_pc_cmax_ema.item())

            if self._cat_sb_enabled:
                state_bound_costs = trajectories["state_bound_costs"].to(**self.tensor_args)
                state_bound_violation = trajectories["state_bound_violation"].to(**self.tensor_args)
                if all_costs.shape != state_bound_costs.shape or all_costs.shape != state_bound_violation.shape:
                    raise RuntimeError(
                        "Shape mismatch for shifted-CaT state bound inputs: "
                        f"costs={tuple(all_costs.shape)}, "
                        f"state_bound_costs={tuple(state_bound_costs.shape)}, "
                        f"state_bound_violation={tuple(state_bound_violation.shape)}."
                    )
                task_costs = task_costs - state_bound_costs
                delta_sb = self._cat_build_delta(
                    state_bound_violation,
                    ema_attr="_cat_sb_cmax_ema",
                    tau_c=self._cat_sb_tau_c,
                    p_max=self._cat_sb_p_max,
                    eps=self._cat_sb_eps,
                )
                delta_list.append(delta_sb)
                debug_stats["cat_sb_delta_max"] = float(torch.max(delta_sb).item())
                debug_stats["cat_sb_cmax_ema"] = float(self._cat_sb_cmax_ema.item())

            if self._cat_torque_enabled:
                torque_costs = trajectories['joint_torque_costs'].to(**self.tensor_args)
                torque_violation = trajectories['joint_torque_violation'].to(**self.tensor_args)
                if torque_costs.shape != all_costs.shape or torque_violation.shape != all_costs.shape:
                    raise RuntimeError('CaT joint torque arrays must match [particles,horizon] costs')
                task_costs = task_costs - torque_costs
                delta_torque = self._cat_build_delta(torque_violation,
                    ema_attr='_cat_torque_cmax_ema', tau_c=self._cat_torque_tau_c,
                    p_max=self._cat_torque_p_max, eps=self._cat_torque_eps)
                delta_list.append(delta_torque)
                debug_stats['cat_torque_delta_max'] = float(delta_torque.max().item())
                debug_stats['cat_torque_relative_excess_max'] = float(torque_violation.max().item())
            if self._cat_self_enabled:
                self_costs=trajectories['self_collision_costs'].to(**self.tensor_args)
                self_violation=trajectories['self_collision_violation'].to(**self.tensor_args)
                if self_costs.shape!=all_costs.shape or self_violation.shape!=all_costs.shape:
                    raise RuntimeError('CaT self-collision arrays must match rollout costs')
                task_costs=task_costs-self_costs
                delta_self=self._cat_build_delta(self_violation,ema_attr='_cat_self_cmax_ema',tau_c=self._cat_self_tau_c,p_max=self._cat_self_p_max,eps=1.e-6)
                delta_list.append(delta_self)
                debug_stats['cat_self_delta_max']=float(delta_self.max().item())
            if len(delta_list) == 0:
                raise RuntimeError("Shifted-CaT enabled but no cat hazards were built.")

            delta = delta_list[0]
            for delta_i in delta_list[1:]:
                delta = torch.maximum(delta, delta_i)
            if len(delta_list) > 1:
                delta_pairwise_max = delta_list[0]
                for delta_i in delta_list[1:]:
                    delta_pairwise_max = torch.maximum(delta_pairwise_max, delta_i)
                debug_stats["cat_delta_max_error"] = float(
                    torch.max(torch.abs(delta - delta_pairwise_max)).item()
                )

            survival = torch.cumprod(1.0 - delta, dim=1)

            eps_shift = torch.as_tensor(self._cat_pc_eps, **self.tensor_args)
            batch_bmax = torch.max(task_costs)
            debug_stats["cat_task_cost_max"] = float(batch_bmax.item())
            b = self._cat_reward_baseline(batch_bmax)
            r_shift = b - task_costs
            if torch.min(r_shift) < -1.0e-6:
                raise RuntimeError(
                    "Shifted-CaT produced negative shifted reward below tolerance. "
                    f"min(r_shift)={float(torch.min(r_shift).item()):.6e}."
                )
            r_shift = torch.clamp(r_shift, min=0.0)

            cat_costs = -survival * r_shift
            traj_costs = cost_to_go(cat_costs, self.gamma_seq)[:,0]

            debug_stats["cat_delta_max"] = float(torch.max(delta).item())
            debug_stats["cat_survival_final_mean"] = float(torch.mean(survival[:, -1]).item())
            debug_stats["cat_pc_survival_final_mean"] = debug_stats["cat_survival_final_mean"]
            debug_stats["cat_pc_bmax_ema"] = float(self._cat_pc_bmax_ema.item())
            self._cat_pc_debug = debug_stats
        control_costs = self._control_costs(actions)

        total_costs = traj_costs + self.beta * control_costs

        
        # #calculate soft-max
        w = torch.softmax((-1.0/self.beta) * total_costs, dim=0)
        self.total_costs = total_costs
        return w

    def _control_costs(self, actions):
        if self.alpha == 1:
            # if not self.time_based_weights:
            return torch.zeros(actions.shape[0], **self.tensor_args)
        else:
            # u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
            # control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
            # control_costs = np.sum(control_costs, axis=-1)
            # control_costs = cost_to_go(control_costs, self.gamma_seq)
            # # if not self.time_based_weights: control_costs = control_costs[:,0]
            # control_costs = control_costs[:,0]
            delta = actions - self.mean_action.unsqueeze(0)
            u_normalized = self.mean_action.matmul(self.full_inv_cov).unsqueeze(0)
            control_costs = 0.5 * u_normalized * (self.mean_action.unsqueeze(0) + 2.0 * delta)
            control_costs = torch.sum(control_costs, dim=-1)
            control_costs = cost_to_go(control_costs, self.gamma_seq)
            control_costs = control_costs[:,0]
        return control_costs
    
    def _calc_val(self, trajectories):
        costs = trajectories["costs"].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        delta = actions - self.mean_action.unsqueeze(0)
        
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.beta * control_costs
        # calculate log-sum-exp
        # c = (-1.0/self.beta) * total_costs.copy()
        # cmax = np.max(c)
        # c -= cmax
        # c = np.exp(c)
        # val1 = cmax + np.log(np.sum(c)) - np.log(c.shape[0])
        # val1 = -self.beta * val1

        # val = -self.beta * scipy.special.logsumexp((-1.0/self.beta) * total_costs, b=(1.0/total_costs.shape[0]))
        val = -self.beta * torch.logsumexp((-1.0/self.beta) * total_costs)
        return val
        
