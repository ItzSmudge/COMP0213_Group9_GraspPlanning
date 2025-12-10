from typing import List
import pybullet as p

from .AbstractGripper import AbstractGripper


class TwoFingerGripper(AbstractGripper):
    """PR2 2-finger gripper."""

    def __init__(self, urdf_path: str = "pr2_gripper.urdf",
                 base_position: List[float] = [0.0, 0.0, 0.5]):
        super().__init__(urdf_path, base_position)

    def open(self) -> None:
        for joint in [0, 2]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.55, force=50
            )

    def close(self) -> None:
        for joint in [0, 2]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.0, force=300
            )

    def weak_close(self) -> None:
        """Close the gripper with a weaker force (used for maintaining grasp)."""
        for joint in [0, 2]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.0, force=150
            )

    def get_grasp_offset(self) -> float:
        return 0.15
