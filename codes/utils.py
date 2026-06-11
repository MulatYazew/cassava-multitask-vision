"""
Utility functions for AgroVision project.
Includes seed management, helper functions, and common utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(*paths):
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Variable number of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_device():
    """
    Get the appropriate device (GPU or CPU).
    
    Returns:
        torch.device: Device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """
    Format seconds to HH:MM:SS format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
