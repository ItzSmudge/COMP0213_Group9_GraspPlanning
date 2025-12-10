from typing import List
import pybullet as p

from .AbstractGripper import AbstractGripper


class ThreeFingerGripper(AbstractGripper):
    """3-finger SDH gripper."""

    def __init__(self, urdf_path: str,
                 base_position: List[float] = [0.0, 0.0, 0.1]):
        super().__init__(urdf_path, base_position)

    def open(self) -> None:
        for base_joint in [1, 4]:
            p.setJointMotorControl2(
                self.id, base_joint, p.POSITION_CONTROL,
                targetPosition=-0.5, force=50
            )
        for tip_joint in [2, 5]:
            p.setJointMotorControl2(
                self.id, tip_joint, p.POSITION_CONTROL,
                targetPosition=0.1, force=50
            )
        p.setJointMotorControl2(
            self.id, 7, p.POSITION_CONTROL,
            targetPosition=-0.5, force=50
        )
        p.setJointMotorControl2(
            self.id, 8, p.POSITION_CONTROL,
            targetPosition=0.1, force=50
        )

    def close(self) -> None:
        for i, target in [
            (0, 1.2), (1, -0.12), (2, 0.8),
            (3, 1.2), (4, -0.12), (5, 0.8),
            (6, 0.0), (7, -0.12), (8, 0.8)
        ]:
            p.setJointMotorControl2(
                self.id, i, p.POSITION_CONTROL,
                targetPosition=target, force=300
            )

    def set_default_position(self) -> None:
        """Set gripper to default pre-grasp configuration."""
        for joint, position in [
            (0, 1.2), (1, -0.2), (2, 0.55),
            (3, 1.2), (4, -0.2), (5, 0.55),
            (6, 0.0), (7, -0.2), (8, 0.55)
        ]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=position, force=50
            )

    def get_grasp_offset(self) -> float:
        return 0.12
