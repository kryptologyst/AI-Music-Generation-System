"""Utility functions for the music generation system."""

import torch
import numpy as np
import random
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def set_deterministic(seed: int = 42) -> None:
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set deterministic behavior with seed {seed}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def create_directory_structure(base_dir: str) -> None:
    """Create the standard directory structure."""
    base_path = Path(base_dir)
    
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "assets",
        "assets/generated_samples",
        "configs/local",
        "tests"
    ]
    
    for dir_path in directories:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {base_dir}")


def save_config_summary(config: Dict[str, Any], output_path: str) -> None:
    """Save a human-readable configuration summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Music Generation System - Configuration Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Model configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Type: {config['model']['model_type']}\n")
        f.write(f"  Hidden Size: {config['model']['hidden_size']}\n")
        f.write(f"  Layers: {config['model']['num_layers']}\n")
        f.write(f"  Dropout: {config['model']['dropout']}\n")
        f.write(f"  Attention: {config['model']['attention']}\n\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  Epochs: {config['training']['num_epochs']}\n")
        f.write(f"  Learning Rate: {config['training']['learning_rate']}\n")
        f.write(f"  Batch Size: {config['data']['batch_size']}\n")
        f.write(f"  Scheduler: {config['training']['scheduler']}\n")
        f.write(f"  Mixed Precision: {config['training']['mixed_precision']}\n\n")
        
        # Generation configuration
        f.write("GENERATION CONFIGURATION:\n")
        f.write(f"  Temperature: {config['generation']['temperature']}\n")
        f.write(f"  Top-k: {config['generation']['top_k']}\n")
        f.write(f"  Top-p: {config['generation']['top_p']}\n")
        f.write(f"  Sampling Strategy: {config['generation']['sampling_strategy']}\n")
        f.write(f"  Max Length: {config['generation']['max_length']}\n\n")
        
        # System configuration
        f.write("SYSTEM CONFIGURATION:\n")
        f.write(f"  Device: {config['system']['device']}\n")
        f.write(f"  Seed: {config['system']['seed']}\n")
        f.write(f"  Deterministic: {config['system']['deterministic']}\n")
        f.write(f"  Log Level: {config['system']['log_level']}\n")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    try:
        # Validate model config
        assert config['model']['model_type'] in ['lstm', 'gru', 'transformer']
        assert config['model']['hidden_size'] > 0
        assert config['model']['num_layers'] > 0
        assert 0 <= config['model']['dropout'] <= 1
        
        # Validate training config
        assert config['training']['num_epochs'] > 0
        assert config['training']['learning_rate'] > 0
        assert config['data']['batch_size'] > 0
        
        # Validate generation config
        assert config['generation']['temperature'] > 0
        assert config['generation']['top_k'] > 0
        assert 0 < config['generation']['top_p'] <= 1
        assert config['generation']['sampling_strategy'] in ['greedy', 'nucleus', 'top_k', 'temperature']
        
        logger.info("Configuration validation passed")
        return True
        
    except (AssertionError, KeyError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_model_size_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get detailed model size information."""
    total_params = count_parameters(model)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        "total_parameters": total_params,
        "formatted_parameters": format_number(total_params),
        "model_size_mb": round(model_size_mb, 2),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)
    }


def log_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Log detailed model information."""
    info = get_model_size_info(model)
    
    logger.info(f"{model_name} Information:")
    logger.info(f"  Total Parameters: {info['formatted_parameters']}")
    logger.info(f"  Model Size: {info['model_size_mb']} MB")
    logger.info(f"  Trainable Parameters: {format_number(info['trainable_parameters'])}")
    logger.info(f"  Non-trainable Parameters: {format_number(info['non_trainable_parameters'])}")


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5) -> None:
    """Clean up old checkpoint files, keeping only the most recent ones."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return
    
    checkpoint_files = list(checkpoint_path.glob("*.ckpt"))
    
    if len(checkpoint_files) <= keep_last:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old checkpoints
    files_to_remove = checkpoint_files[keep_last:]
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            logger.info(f"Removed old checkpoint: {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to remove {file_path.name}: {e}")


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    samples_per_second: float = 100.0
) -> Dict[str, float]:
    """Estimate training time based on dataset size and model complexity."""
    batches_per_epoch = num_samples / batch_size
    total_batches = batches_per_epoch * num_epochs
    
    # Estimate time per batch (rough estimate)
    time_per_batch = batch_size / samples_per_second
    
    total_time_seconds = total_batches * time_per_batch
    
    return {
        "total_time_seconds": total_time_seconds,
        "total_time_minutes": total_time_seconds / 60,
        "total_time_hours": total_time_seconds / 3600,
        "batches_per_epoch": batches_per_epoch,
        "total_batches": total_batches
    }
