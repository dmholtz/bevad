from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    """Base class for all transforms (observation/action).

    Defines the interface for transform classes; subclasses must implement
    __call__ to apply the transformation to the data.
    """

    def __init__(self):
        """Initialize the BaseTransform."""
        pass

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply the transform to the given data.

        Args:
            data: Input data to be transformed.

        Returns:
            Transformed data.
        """
        pass
