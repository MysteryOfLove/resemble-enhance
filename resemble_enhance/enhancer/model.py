"""
Model definitions for inference without training dependencies.
This module provides the same models as train.py but without DeepSpeed dependencies.
"""
import logging
from pathlib import Path

import torch

from .enhancer_inference import EnhancerInference
from .hparams import HParams

logger = logging.getLogger(__name__)


def load_enhancer_model(run_dir: Path, device: str = "cpu") -> EnhancerInference:
    """Load enhancer model for inference without DeepSpeed dependencies.
    
    Args:
        run_dir: Path to the model checkpoint directory
        device: Device to load the model on
        
    Returns:
        Loaded Enhancer model ready for inference
    """
    hp = HParams.load(run_dir)
    model = EnhancerInference(hp)
    
    # Load the state dict from DeepSpeed checkpoint
    ckpt_path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location="cpu")["module"]
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model


def create_default_enhancer(device: str = "cpu") -> EnhancerInference:
    """Create a default enhancer model with default hyperparameters.
    
    Args:
        device: Device to create the model on
        
    Returns:
        Default Enhancer model (not trained)
    """
    hp = HParams()
    model = EnhancerInference(hp)
    model.eval()
    model.to(device)
    return model
