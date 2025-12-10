from typing import List, Optional
import os
import numpy as np
import pybullet as p
import pybullet_data

from grippers.AbstractGripper import AbstractGripper
from grippers.TwoFingerGripper import TwoFingerGripper
from objects.AbstractObject import AbstractObject


class GraspSimulator:
    """Manage PyBullet simulation and execute grasps."""

    def __init__(self, gui: bool = True, time_step: float = 1.0 / 240.0):
        self.gui = gui
        self.time_step = time_step
        self.plane_id: Optional[int] = None
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return

        mode = p.GUI if self.gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        #urdf_path = os.path.join(project_root, "urdf")
        # p.setAdditionalSearchPath(urdf_path)

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSubSteps=2)

        self.plane_id = p.loadURDF("plane.urdf")

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0.1]
            )

        self.connected = True
        print("PyBullet environment initialized (balanced physics mode)")

    def disconnect(self) -> None:
        """ stop communication with the pybullet simulation """
        if self.connected:
            p.disconnect()
            self.connected = False

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation."""
        for _ in range(n_steps):
            p.stepSimulation()

    def execute_grasp(
        self,
        gripper: AbstractGripper,
        obj: AbstractObject,
        position: List[float],
        orientation: List[float],
        lift_height: float = 0.6
    ) -> bool:
        """Execute a grasp and report success."""

        is_cylinder = obj.is_unstable()

        # Remove and reload object
        obj.remove()
        self.step(2)

        gripper.move_to_pose(position, orientation)
        self.step(5)

        # Open gripper and load object
        obj.load()
        self.step(2)
        gripper.open()
        self.step(2)

        # Approach the object
        start_pos = np.array(position)
        target_pos = np.array(obj.position) + \
            np.array([0, 0, obj.get_grasp_height()])

        approach_factor = 3.67 if isinstance(
            gripper, TwoFingerGripper) else 1.6
        approach_steps = 25 if is_cylinder else 15

        for t in range(approach_steps):
            alpha = (t + 1) / approach_steps
            new_pos = start_pos + (alpha / approach_factor) * \
                (target_pos - start_pos)
            gripper.move_to_pose(new_pos.tolist(), orientation)
            self.step(1)

        # For unstable (cylindrical) objects, partial close then settle
        if is_cylinder and isinstance(gripper, TwoFingerGripper):
            for joint in [0, 2]:
                p.setJointMotorControl2(
                    gripper.id, joint, p.POSITION_CONTROL,
                    targetPosition=0.2, force=100
                )
            self.step(10)

        # Full close
        gripper.close()
        self.step(10 if is_cylinder else 5)

        # Lift
        lift_start = new_pos
        lift_end = np.array([new_pos[0], new_pos[1], lift_height])
        lift_steps = 25 if is_cylinder else 15

        for i in range(lift_steps):
            alpha = i / lift_steps
            interp_pos = lift_start * (1 - alpha) + lift_end * alpha
            gripper.move_to_pose(interp_pos.tolist(), orientation)

            if isinstance(gripper, TwoFingerGripper):
                gripper.weak_close()
            else:
                gripper.close()

            p.stepSimulation()

        success = self._check_grasp_success(obj, gripper, is_cylinder)
        self.step(5)

        return success

    def _check_grasp_success(
        self,
        obj: AbstractObject,
        gripper: AbstractGripper,
        is_cylinder: bool = False
    ) -> bool:
        """ check whether the object has remained in the gripper and hasnt slipped """
        if isinstance(gripper, TwoFingerGripper):
            gripper.weak_close()
        else:
            gripper.close()

        hold_steps = 50 if is_cylinder else 30
        self.step(hold_steps)

        obj_pos, _ = p.getBasePositionAndOrientation(obj.id)
        height = obj_pos[2]

        threshold = 0.20 if is_cylinder else 0.25
        success = height > threshold

        if success:
            print(f"Success (height: {height:.3f})")
        else:
            print(f"Failure (height: {height:.3f})")

        return success
