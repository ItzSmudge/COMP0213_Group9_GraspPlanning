from typing import List, Tuple

from .AbstractObject import AbstractObject


class CylinderObject(AbstractObject):
    """Cylinder object."""

    def __init__(self, position: List[float], urdf_path: str):
        super().__init__(urdf_path, position)

    def get_grasp_height(self) -> float:
        return 0.1

    def get_bounding_box(self) -> Tuple[float, float, float]:
        return 0.06, 0.06, 0.2

    def is_unstable(self) -> bool:
        return True
