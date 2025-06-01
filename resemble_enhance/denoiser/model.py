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


def load_denoiser_from_enhancer_checkpoint(run_dir: Path | None, device: str = "cpu") -> Denoiser:
    """Load denoiser model from an enhancer checkpoint.
    
    This extracts the denoiser weights from an enhancer checkpoint that contains
    both enhancer and denoiser weights.
    
    Args:
        run_dir: Path to the enhancer checkpoint directory (None for default model)
        device: Device to load the model on
        
    Returns:
        Loaded Denoiser model ready for inference
    """
    # If no run_dir provided, create default model
    if run_dir is None:
        return create_default_denoiser(device)
    
    run_dir = Path(run_dir)
    
    # Try to load denoiser hparams first, fall back to enhancer hparams
    denoiser_hp_path = run_dir / "denoiser_hparams.yaml"
    if denoiser_hp_path.exists():
        hp = HParams.load(denoiser_hp_path)
    else:
        # Load enhancer hparams and use denoiser settings from it
        from ..enhancer.hparams import HParams as EnhancerHParams
        enhancer_hp_path = run_dir / "hparams.yaml"
        if enhancer_hp_path.exists():
            enhancer_hp = EnhancerHParams.load(run_dir)
            
            # Create denoiser hparams from enhancer config
            hp = HParams()
            # Copy relevant settings if they exist
            if hasattr(enhancer_hp, 'denoiser_run_dir') and enhancer_hp.denoiser_run_dir:
                denoiser_run_dir = Path(enhancer_hp.denoiser_run_dir)
                if (denoiser_run_dir / "hparams.yaml").exists():
                    hp = HParams.load(denoiser_run_dir)
        else:
            # No hparams found, use default
            hp = HParams()
    
    model = Denoiser(hp)
    
    # Load the state dict from enhancer checkpoint
    ckpt_path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    if not ckpt_path.exists():
        # No checkpoint found, return default model
        return create_default_denoiser(device)
    
    state_dict = torch.load(ckpt_path, map_location="cpu")["module"]
    
    # Extract only denoiser weights
    denoiser_state_dict = {k.replace('denoiser.', '', 1): v for k, v in state_dict.items() if k.startswith('denoiser.')}
    
    if not denoiser_state_dict:
        # No denoiser weights found, return default model
        return create_default_denoiser(device)
    
    model.load_state_dict(denoiser_state_dict)
    model.eval()
    model.to(device)
    
    return model
