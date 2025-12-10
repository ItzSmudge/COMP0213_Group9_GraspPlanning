from typing import List, Tuple

import numpy as np

from .AbstractSampler import AbstractSampler
from objects.AbstractObject import AbstractObject
from objects.CylinderObject import CylinderObject


class SphericalSampler(AbstractSampler):
    """
    Generate grasp samples by placing the gripper on a sphere around the object
    with random initial position on the sphere surface.
    """

    def __init__(
        self,
        radius: float = 0.4,
        noise_std: float = 0.03,
        phi_max: float = 0.6 * np.pi
    ):
        self.radius = radius
        self.noise_std = noise_std
        self.phi_max = phi_max
        self._obj_center_cache = None

    def sample_pose(
        self,
        object_position: np.ndarray,
        object_type: AbstractObject
    ) -> Tuple[List[float], List[float]]:
        """Sample a single grasp pose."""
        obj_center = object_position + np.array([0, 0, -0.025])

        # Point on sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, self.phi_max)
        r = self.radius + np.random.uniform(-self.noise_std, self.noise_std)

        sin_phi = np.sin(phi)
        position = obj_center + r * np.array([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            np.cos(phi)
        ])

        # Orientation pointing towards object
        direction = obj_center - position
        direction = direction / np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0])
        horizontal_dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        pitch = np.arctan2(-direction[2], horizontal_dist)
        roll = np.random.uniform(-0.2, 0.2)

        # Adjustment for cylinder to improve success rate
        if isinstance(object_type, CylinderObject):
            pitch += np.pi / 2

        return position.tolist(), [roll, pitch, yaw]

    def sample_multiple(
        self,
        n_samples: int,
        object_position: np.ndarray,
        object_type: AbstractObject
    ) -> list[Tuple[List[float], List[float]]]:
        return [self.sample_pose(object_position, object_type) for _ in range(n_samples)]
