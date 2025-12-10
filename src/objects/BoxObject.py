from typing import List, Tuple

from .AbstractObject import AbstractObject


class BoxObject(AbstractObject):
    """Box object."""

    def __init__(self, position: List[float]):
        super().__init__("cube_small.urdf", position)

    def get_grasp_height(self) -> float:
        return 0.03

    def get_bounding_box(self) -> Tuple[float, float, float]:
        return 0.05, 0.05, 0.05

    def is_unstable(self) -> bool:
        return False  # Boxes are stable
