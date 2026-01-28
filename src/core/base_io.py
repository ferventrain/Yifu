from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Optional
from pathlib import Path
import numpy as np

class BaseIO(ABC):
    """
    Abstract base class for Input/Output operations.
    Handles reading raw data (TIFF/Zarr) and writing results.
    """
    
    @abstractmethod
    def load_image(self, path: Union[str, Path]) -> object:
        """
        Load an image or volume.
        Should return an array-like object (numpy array, zarr array, or lazy loader).
        """
        pass

    @abstractmethod
    def save_image(self, data: object, path: Union[str, Path], **kwargs):
        """
        Save an image or volume to disk.
        """
        pass
    
    @abstractmethod
    def get_metadata(self, path: Union[str, Path]) -> dict:
        """
        Retrieve metadata (resolution, shape, dtype) without loading full data.
        """
        pass
