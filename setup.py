#!/usr/bin/env python3
"""Setup script for the music generation system."""

import os
import sys
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def setup_environment():
    """Setup the development environment."""
    logger.info("Setting up music generation system environment...")
    
    # Create directory structure
    logger.info("Creating directory structure...")
    directories = [
        "data/raw",
        "data/processed",
        "checkpoints", 
        "logs",
        "assets",
        "assets/generated_samples",
        "configs/local"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Install dependencies
    logger.info("Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        logger.error("Failed to install dependencies")
        return False
    
    # Install pre-commit hooks (optional)
    if Path(".pre-commit-config.yaml").exists():
        logger.info("Installing pre-commit hooks...")
        run_command("pre-commit install", "Installing pre-commit hooks")
    
    # Generate toy dataset
    logger.info("Generating toy dataset...")
    try:
        from src.data.processor import generate_toy_dataset
        generate_toy_dataset("data/raw", num_files=20)
        logger.info("✓ Toy dataset generated successfully")
    except Exception as e:
        logger.warning(f"Failed to generate toy dataset: {e}")
    
    logger.info("✓ Environment setup completed successfully!")
    return True


def verify_installation():
    """Verify that the installation is working correctly."""
    logger.info("Verifying installation...")
    
    try:
        # Test imports
        import torch
        import music21
        import streamlit
        logger.info("✓ All required packages imported successfully")
        
        # Test device detection
        from src.training.trainer import get_device
        device = get_device()
        logger.info(f"✓ Device detection working: {device}")
        
        # Test model creation
        from src.models.architecture import create_model
        config = {
            "model_type": "lstm",
            "input_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 128
        }
        model = create_model(config)
        logger.info("✓ Model creation working")
        
        logger.info("✓ Installation verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Installation verification failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Music Generation System Setup")
    logger.info("=" * 40)
    
    if not setup_environment():
        logger.error("Setup failed!")
        sys.exit(1)
    
    if not verify_installation():
        logger.error("Installation verification failed!")
        sys.exit(1)
    
    logger.info("\n" + "=" * 40)
    logger.info("Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Train a model: python scripts/train.py")
    logger.info("2. Generate music: python scripts/generate.py")
    logger.info("3. Launch demo: streamlit run demo/app.py")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
