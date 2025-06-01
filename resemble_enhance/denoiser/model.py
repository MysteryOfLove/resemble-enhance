"""
Model definitions for denoiser inference without training dependencies.
This module provides the same models as train.py but without DeepSpeed dependencies.
"""
import logging
from pathlib import Path

import torch

from .denoiser import Denoiser
from .hparams import HParams

logger = logging.getLogger(__name__)


def load_denoiser_model(run_dir: Path, device: str = "cpu") -> Denoiser:
    """Load denoiser model for inference without DeepSpeed dependencies.
    
    Args:
        run_dir: Path to the model checkpoint directory
        device: Device to load the model on
        
    Returns:
        Loaded Denoiser model ready for inference
    """
    hp = HParams.load(run_dir)
    model = Denoiser(hp)
    
    # Load the state dict from DeepSpeed checkpoint
    ckpt_path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location="cpu")["module"]
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model


def create_default_denoiser(device: str = "cpu") -> Denoiser:
    """Create a default denoiser model with default hyperparameters.
    
    Args:
        device: Device to create the model on
        
    Returns:
        Default Denoiser model (not trained)
    """
    hp = HParams()
    model = Denoiser(hp)
    model.eval()
    model.to(device)
    return model
