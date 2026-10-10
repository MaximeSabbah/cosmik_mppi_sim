"""Batched RNEA for a fixed-base seven-revolute-joint serial arm.

Pinocchio parses and aggregates fixed-link inertias once. The GPU recurrence uses
Pinocchio's linear/angular spatial-vector ordering and includes gravity. No
collision-only link is treated as a dynamic body. Predictions are nominal rigid
body torques, without feedback tracking corrections or unmodelled friction.
The default matrix backend preserves the original arithmetic. The opt-in fused
backend expands that same recurrence for torch.compile kernel fusion.
"""
import numpy as np
import torch
import pinocchio as pin


class BatchedInverseDynamics(torch.nn.Module):
    def __init__(self, urdf_path, tensor_args, compile_model=True, backend='matrix'):
        super().__init__()
        if backend not in ('matrix', 'fused'):
            raise ValueError("RNEA backend must be 'matrix' or 'fused'")
        self.backend = backend
        model = pin.buildModelFromUrdf(str(urdf_path))
        if model.nq != 7 or model.nv != 7 or model.njoints != 8:
            raise ValueError('Torque rollout requires a fixed-base seven-DOF reduced arm URDF')
        if any(model.parents[i] != i-1 or model.joints[i].shortname() != 'JointModelRZ'
               for i in range(1, 8)):
            raise ValueError('Torque rollout currently supports a serial chain of Z revolute joints')
        for name, value in dict(
            fixed_rotation=np.array([x.rotation for x in model.jointPlacements[1:]]),
            translation=np.array([x.translation for x in model.jointPlacements[1:]]),
            inertia=np.array([x.matrix() for x in model.inertias[1:]]),
            gravity=model.gravity.vector.copy(),
            effort_limits=model.effortLimit.copy(),
        ).items():
            self.register_buffer(name, torch.as_tensor(value, **tensor_args))
        self.compile_model = bool(compile_model)
        self._evaluate = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_evaluate"] = None  # Compile in the optimizer subprocess after deserialization.
        return state

    def forward(self, q, qd, qdd):
        if q.shape != qd.shape or q.shape != qdd.shape or q.shape[-1] != 7:
            raise ValueError('RNEA expects matching (...,7) position, velocity and acceleration')
        if self._evaluate is None:
            # Missing backend on an older pickled model keeps its original path.
            evaluate = self._rnea_fused if getattr(self, 'backend', 'matrix') == 'fused' else self._rnea
            self._evaluate = torch.compile(evaluate, fullgraph=True, dynamic=False) if self.compile_model else evaluate
        # STORM wraps optimization in autocast. RNEA reductions must keep a
        # consistent dtype and full float32 precision for small wrist torques.
        with torch.autocast(device_type=q.device.type, enabled=False):
            return self._evaluate(q.reshape(-1,7).to(self.inertia.dtype),
                                  qd.reshape(-1,7).to(self.inertia.dtype),
                                  qdd.reshape(-1,7).to(self.inertia.dtype)).reshape(q.shape)

    def _rnea_fused(self, q, qd, qdd):
        from .fused_rnea import rnea_fused
        return rnea_fused(self, q, qd, qdd)

    def _rnea(self, q, qd, qdd):
        zero = torch.zeros_like(q[:,0]); one = torch.ones_like(zero)
        velocity = torch.zeros_like(q[:,:6]); acceleration = -self.gravity.expand(q.shape[0],6)
        forces = []; rotations = []
        for i in range(7):
            c = torch.cos(q[:,i]); s = torch.sin(q[:,i])
            rz = torch.stack((c,-s,zero,s,c,zero,zero,zero,one),-1).reshape(-1,3,3)
            rot = self.fixed_rotation[i] @ rz; rt = rot.transpose(-1,-2)
            p = self.translation[i].expand(q.shape[0],3)
            parent_v, parent_w = velocity[:,:3], velocity[:,3:]
            linear_v = (rt @ (parent_v-torch.cross(p,parent_w,dim=-1)).unsqueeze(-1)).squeeze(-1)
            joint_w = torch.stack((zero,zero,qd[:,i]),-1)
            angular_v = (rt @ parent_w.unsqueeze(-1)).squeeze(-1) + joint_w
            linear_a = (rt @ (acceleration[:,:3]-torch.cross(p,acceleration[:,3:],dim=-1)).unsqueeze(-1)).squeeze(-1) + torch.cross(linear_v,joint_w,dim=-1)
            angular_a = (rt @ acceleration[:,3:].unsqueeze(-1)).squeeze(-1) + torch.stack((zero,zero,qdd[:,i]),-1) + torch.cross(angular_v,joint_w,dim=-1)
            velocity = torch.cat((linear_v,angular_v),-1); acceleration = torch.cat((linear_a,angular_a),-1)
            momentum = velocity @ self.inertia[i].T
            bias = torch.cat((torch.cross(angular_v,momentum[:,:3],dim=-1),torch.cross(linear_v,momentum[:,:3],dim=-1)+torch.cross(angular_v,momentum[:,3:],dim=-1)),-1)
            forces.append(acceleration @ self.inertia[i].T + bias); rotations.append(rot)
        torques=[]
        for i in range(6,-1,-1):
            f=forces[i];torques.append(f[:,5])
            if i:
                linear_f=(rotations[i] @ f[:,:3].unsqueeze(-1)).squeeze(-1)
                angular_f=(rotations[i] @ f[:,3:].unsqueeze(-1)).squeeze(-1)+torch.cross(self.translation[i].expand(q.shape[0],3),linear_f,dim=-1)
                forces[i-1]=forces[i-1]+torch.cat((linear_f,angular_f),-1)
        return torch.stack(torques[::-1],-1)
