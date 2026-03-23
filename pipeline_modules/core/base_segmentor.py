from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Dict
import numpy as np

class BaseSegmentor(ABC):
    """
    Abstract base class for 3D segmentation models.
    Designed to be pluggable (Cellpose, Spotiflow, etc.).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the segmentor with a configuration dictionary.
        Config should contain model path, parameters, device settings, etc.
        """
        self.config = config

    @abstractmethod
    def load_model(self):
        """
        Load the model weights into memory/GPU.
        """
        pass

    @abstractmethod
    def predict_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of data.
        
        Args:
            batch_data: Input array (e.g., [Batch, Z, Y, X] or [Batch, C, Z, Y, X])
            
        Returns:
            Output mask/probability map.
        """
        pass
    
    @abstractmethod
    def predict_volume(self, volume_data: object) -> object:
        """
        Run inference on a large volume (usually implementing sliding window or block processing).
        """
        pass
