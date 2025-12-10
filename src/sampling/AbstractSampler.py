from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from objects.AbstractObject import AbstractObject


class AbstractSampler(ABC):
    """Abstract base class for grasp pose samplers."""

    @abstractmethod
    def sample_pose(
        self,
        object_position: np.ndarray,
        object_type: AbstractObject
    ) -> Tuple[List[float], List[float]]:
        """Sample one grasp pose."""
        pass

    @abstractmethod
    def sample_multiple(
        self,
        n_samples: int,
        object_position: np.ndarray,
        object_type: AbstractObject
    ) -> list[Tuple[List[float], List[float]]]:
        """Sample multiple grasp poses."""
        pass
