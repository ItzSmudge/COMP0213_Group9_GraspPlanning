from abc import ABC, abstractmethod
from typing import List, Optional
import pybullet as p


class AbstractGripper(ABC):
    """Abstract base class for all grippers."""

    def __init__(self, urdf_path: str, base_position: List[float]):
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.id: Optional[int] = None
        self.constraint_id: Optional[int] = None

    def load(self) -> int:
        """Load the gripper URDF into the simulation."""
        self.id = p.loadURDF(
            self.urdf_path,
            self.base_position,
            useFixedBase=False
        )
        return self.id

    def attach_fixed(self, offset: List[float] = [0, 0, 0]) -> None:
        """Attach to a fixed constraint for controlled movement of the gripper."""
        if self.id is None:
            raise ValueError("Gripper must be loaded before attaching!")

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.base_position
        )

    def move_to_pose(self, position: List[float], orientation: List[float]) -> None:
        """Move the gripper to a specified pose."""
        if self.constraint_id is None:
            raise ValueError("Gripper has not been attached")

        quat = p.getQuaternionFromEuler(orientation)
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=position,
            jointChildFrameOrientation=quat,
            maxForce=500
        )

    @abstractmethod
    def open(self) -> None:
        """Open gripper."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close gripper."""
        pass

    @abstractmethod
    def get_grasp_offset(self) -> float:
        """Return the ideal approach distance."""
        pass
