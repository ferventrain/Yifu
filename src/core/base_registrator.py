from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any
from pathlib import Path
import numpy as np

class BaseRegistrator(ABC):
    """
    Abstract base class for 3D image registration.
    Handles Atlas <-> Image registration logic.
    """

    def __init__(self, config: Dict):
        """
        Initialize registrator with configuration.
        Config should include atlas paths, registration parameters (rigid/affine/syn), etc.
        """
        self.config = config

    @abstractmethod
    def load_atlas(self, atlas_path: Union[str, Path], label_path: Union[str, Path]):
        """
        Load reference atlas and annotation labels.
        """
        pass

    @abstractmethod
    def register_atlas_to_image(self, target_image: object) -> Dict[str, Any]:
        """
        Compute transform to map Atlas onto Target Image.
        
        Args:
            target_image: The sample brain image (usually downsampled).
            
        Returns:
            Dictionary containing:
            - 'warped_atlas': The atlas warped to sample space
            - 'warped_labels': The labels warped to sample space
            - 'forward_transforms': List of transform files
            - 'inverse_transforms': List of inverse transform files
        """
        pass

    @abstractmethod
    def register_image_to_atlas(self, moving_image: object) -> Dict[str, Any]:
        """
        Compute transform to map Moving Image onto Atlas (Standard Space).
        """
        pass
    
    @abstractmethod
    def apply_transform(self, image: object, transform_list: list, interpolator: str = 'linear') -> object:
        """
        Apply a computed transform to a new image (e.g., applying registration result to a full-res density map).
        """
        pass
