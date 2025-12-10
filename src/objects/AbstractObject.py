from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import pybullet as p


class AbstractObject(ABC):
    """Abstract base class for all generated objects."""

    def __init__(
        self,
        urdf_file: str,
        position: List[float],
        orientation: Tuple[float, float, float] = (0, 0, 0)
    ):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id: Optional[int] = None

    def load(self) -> int:
        """Load the object into the simulation window."""
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id

    def remove(self) -> None:
        """Remove object from the simulation window."""
        if self.id is not None:
            p.removeBody(self.id)
            self.id = None

    @abstractmethod
    def get_grasp_height(self) -> float:
        """Return the offset value for the ideal height for grasping an object."""
        pass

    @abstractmethod
    def get_bounding_box(self) -> Tuple[float, float, float]:
        """Return the dimensions of the object (height, width, depth)."""
        pass

    @abstractmethod
    def is_unstable(self) -> bool:
        """Return True if the object is prone to slipping."""
        pass
