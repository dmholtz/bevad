from __future__ import annotations

from typing import Any, Callable

import numpy as np

from bevad_sim.data_interface.transforms.base_transform import BaseTransform


class Compose(BaseTransform):
    """Compose multiple transforms into a single sequential transform.

    This class allows chaining multiple transforms together, where each transform
    is applied sequentially to the input data. The data is passed through each transform
    in the order they are specified in the `transforms` list.
    """

    def __init__(self, transforms: list[Callable]):
        """Initialize a Compose object with a sequence of transforms.

        Args:
            transforms (list[Callable]): A list of transforms (callable objects) to
                                          be applied sequentially on the input data.
        """
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        """Apply each transform sequentially to the input data.

        Args:
            data (Any): The input data to be transformed.

        Returns:
            Any: The transformed data after applying all the transforms.

        Notes:
            Each transform in the `transforms` list will be applied to the data
            in order. The output of one transform becomes the input to the next.
        """
        for transform in self.transforms:
            data = transform(data)
        return data


def invert_transform(m: np.ndarray):
    """
    Inverts a stack of 4x4 homogeneous transformation matrices.
    This function computes the inverse of a transformation matrix that represents
    a rigid body transformation in 3D space (rotation and translation). The input
    matrices is assumed to be a ...,4x4 NumPy array with the upper-left 3x3 block as the
    rotation matrix and the upper-right 3x1 vector as the translation.

    Args:
        m (np.ndarray): A ...,4x4 NumPy array representing the transformation matrices.

    Returns:
        np.ndarray: A ...,4x4 NumPy array representing the inverse transformation matrices.

    """
    irot = np.moveaxis(m[..., :3, :3], -1, -2)
    ip = -(irot @ m[..., :3, 3, None]).squeeze(-1)
    res = np.zeros_like(m)
    res[..., :3, :3] = irot
    res[..., :3, 3] = ip
    res[..., 3, 3] = 1.0
    return res
