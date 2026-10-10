"""Scalar-expanded RNEA recurrence suitable for torch.compile kernel fusion.

The rigid-body model, gravity and linear/angular spatial-vector convention are
identical to BatchedInverseDynamics's matrix backend. This only changes how the
same arithmetic is expressed, and uses no benchmark controller dependency.
"""

import torch


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _mat(matrix, vector):
    return tuple(sum(matrix[i, j] * vector[j] for j in range(3)) for i in range(3))


def rnea_fused(model, q, qd, qdd):
    zero = torch.zeros_like(q[:, 0])
    velocity = (zero, zero, zero)
    angular_velocity = (zero, zero, zero)
    acceleration = tuple(-model.gravity[i] + zero for i in range(3))
    angular_acceleration = tuple(-model.gravity[i + 3] + zero for i in range(3))
    forces = []
    angles = []
    for i in range(7):
        cosine = torch.cos(q[:, i])
        sine = torch.sin(q[:, i])
        angles.append((cosine, sine))
        rotation = model.fixed_rotation[i]
        translation = tuple(model.translation[i, j] for j in range(3))

        def rotate_parent(vector):
            vector = _mat(rotation.T, vector)
            return (cosine * vector[0] + sine * vector[1],
                    -sine * vector[0] + cosine * vector[1], vector[2])

        joint_velocity = (zero, zero, qd[:, i])
        joint_acceleration = (zero, zero, qdd[:, i])
        velocity = rotate_parent(_sub(velocity, _cross(translation, angular_velocity)))
        angular_velocity = _add(rotate_parent(angular_velocity), joint_velocity)
        acceleration = _add(
            rotate_parent(_sub(acceleration, _cross(translation, angular_acceleration))),
            _cross(velocity, joint_velocity))
        angular_acceleration = _add(
            _add(rotate_parent(angular_acceleration), joint_acceleration),
            _cross(angular_velocity, joint_velocity))
        spatial_velocity = velocity + angular_velocity
        spatial_acceleration = acceleration + angular_acceleration
        inertia = model.inertia[i]
        momentum = tuple(sum(inertia[j, k] * spatial_velocity[k] for k in range(6)) for j in range(6))
        inertial_force = tuple(sum(inertia[j, k] * spatial_acceleration[k] for k in range(6)) for j in range(6))
        force = _add(inertial_force[:3], _cross(angular_velocity, momentum[:3]))
        moment = _add(inertial_force[3:], _add(_cross(velocity, momentum[:3]),
                                            _cross(angular_velocity, momentum[3:])))
        forces.append((force, moment))
    torques = []
    for i in range(6, -1, -1):
        force, moment = forces[i]
        torques.append(moment[2])
        if i:
            cosine, sine = angles[i]
            rotation = model.fixed_rotation[i]
            translation = tuple(model.translation[i, j] for j in range(3))

            def rotate_child(vector):
                return _mat(rotation, (cosine * vector[0] - sine * vector[1],
                                       sine * vector[0] + cosine * vector[1], vector[2]))

            force = rotate_child(force)
            moment = _add(rotate_child(moment), _cross(translation, force))
            parent_force, parent_moment = forces[i - 1]
            forces[i - 1] = (_add(parent_force, force), _add(parent_moment, moment))
    return torch.stack(torques[::-1], -1)
