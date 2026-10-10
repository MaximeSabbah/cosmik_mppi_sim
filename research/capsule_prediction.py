"""Small, ROS-independent prediction contract for student experiments.

Not connected to MPPI yet. Coordinates are metres in one fixed frame; endpoints
are capsule centre-line ends, not surface points. Times are absolute seconds in
one clock domain. A learned predictor can implement the same Forecast contract.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Observation:
    stamp: float
    frame: str
    ids: tuple[str, ...]
    endpoints: np.ndarray  # [C, 2, 3]
    radii: np.ndarray  # [C]

    def __post_init__(self):
        if not self.frame or not np.isfinite(self.stamp):
            raise ValueError('A finite timestamp and fixed frame are required')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Capsule IDs must be unique')
        ends = np.array(self.endpoints, dtype=np.float64, copy=True)
        radii = np.array(self.radii, dtype=np.float64, copy=True)
        if ends.shape != (len(self.ids), 2, 3) or radii.shape != (len(self.ids),):
            raise ValueError('Expected endpoints [C,2,3] and radii [C]')
        if not np.all(np.isfinite(ends)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError('Coordinates must be finite and radii positive')
        ends.setflags(write=False)
        radii.setflags(write=False)
        object.__setattr__(self, 'endpoints', ends)
        object.__setattr__(self, 'radii', radii)


@dataclass(frozen=True)
class Forecast:
    observation_stamp: float
    frame: str
    ids: tuple[str, ...]
    times: np.ndarray  # [H], absolute rollout evaluation times
    endpoints: np.ndarray  # [H, C, 2, 3]
    radii: np.ndarray  # [C], fixed over this forecast


def predict(current: Observation, times, previous: Observation | None = None) -> Forecast:
    """Zero-order hold, or constant translation velocity from two observations.

    Match by ID, never list position. New IDs hold their latest pose. Capsule
    orientation and length are held fixed. The reference CV model does not infer
    articulation, uncertainty, or a stale-data policy; those belong in the study.
    """
    times = np.array(times, dtype=np.float64, copy=True)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError('Expected nonempty finite rollout times [H]')
    if np.any(np.diff(times) < 0) or np.any(times < current.stamp):
        raise ValueError('Rollout times must be ordered and not precede the observation')
    velocity = np.zeros((len(current.ids), 3))
    if previous is not None:
        if previous.frame != current.frame or previous.stamp >= current.stamp:
            raise ValueError('History must share the frame and strictly precede current')
        old_index = {key: i for i, key in enumerate(previous.ids)}
        for i, key in enumerate(current.ids):
            if key in old_index:
                old = previous.endpoints[old_index[key]].mean(axis=0)
                velocity[i] = (current.endpoints[i].mean(axis=0) - old) / (current.stamp - previous.stamp)
    ends = current.endpoints[None] + (times - current.stamp)[:, None, None, None] * velocity[None, :, None, :]
    return Forecast(current.stamp, current.frame, current.ids, times, ends, current.radii.copy())
