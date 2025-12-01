#!/usr/bin/env python3
"""Main training script for music generation system."""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config, get_default_config
from src.data.processor import MusicDataProcessor, create_data_loaders, generate_toy_dataset
from src.training.trainer import MusicTrainer, get_device
from src.evaluation.metrics import MusicMetrics, create_evaluation_report
from src.generation.sampler import MusicSampler, MusicGenerator, load_model_for_generation

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_data(config):
    """Prepare training data."""
    data_config = config["data"]
    processor = MusicDataProcessor(data_config)
    
    # Check if processed data exists
    processed_dir = Path(data_config["processed_data_dir"])
    if processed_dir.exists() and (processed_dir / "vocabulary.pkl").exists():
        logger.info("Loading existing processed data...")
        train_sequences, val_sequences, test_sequences = processor.load_processed_data(
            data_config["processed_data_dir"]
        )
    else:
        # Check if raw data exists
        raw_dir = Path(data_config["raw_data_dir"])
        if not raw_dir.exists() or not list(raw_dir.glob("*.mid")):
            logger.info("No raw data found, generating toy dataset...")
            generate_toy_dataset(data_config["raw_data_dir"], num_files=20)
        
        logger.info("Processing raw data...")
        all_sequences = processor.process_directory(data_config["raw_data_dir"])
        
        if not all_sequences:
            raise ValueError("No valid sequences found in raw data")
        
        # Create vocabulary
        processor.create_vocabulary(all_sequences)
        
        # Convert to integers
        int_sequences = processor.sequences_to_ints(all_sequences)
        
        # Create splits
        train_sequences, val_sequences, test_sequences = processor.create_splits(
            int_sequences,
            train_split=data_config["train_split"],
            val_split=data_config["val_split"],
            test_split=data_config["test_split"]
        )
        
        # Save processed data
        processor.save_processed_data(
            train_sequences, val_sequences, test_sequences,
            data_config["processed_data_dir"]
        )
    
    # Update config with vocabulary size
    config["model"]["output_size"] = processor.vocab_size
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_sequences, val_sequences, test_sequences, data_config
    )
    
    return train_loader, val_loader, test_loader, processor


def train_model(config, train_loader, val_loader, test_loader):
    """Train the music generation model."""
    trainer = MusicTrainer(config)
    
    # Train the model
    model = trainer.train(train_loader, val_loader, test_loader)
    
    return model


def evaluate_model(model, test_loader, processor, config):
    """Evaluate the trained model."""
    device = get_device()
    metrics_calculator = MusicMetrics(processor.vocab_size)
    
    # Evaluate model
    metrics = metrics_calculator.evaluate_model(
        model, test_loader, device, processor.int_to_note
    )
    
    # Create evaluation report
    report = create_evaluation_report(metrics, f"{config['model']['model_type']} Music Model")
    logger.info(report)
    
    # Save report
    report_path = Path(config["system"]["assets_dir"]) / "evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    
    return metrics


def generate_samples(model, processor, config):
    """Generate sample music pieces."""
    device = get_device()
    
    # Load model for generation
    checkpoint_dir = Path(config["system"]["checkpoint_dir"])
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    
    if not checkpoint_files:
        logger.warning("No checkpoint files found for generation")
        return
    
    # Use the best checkpoint
    best_checkpoint = min(checkpoint_files, key=lambda x: float(x.stem.split('-')[-1]))
    
    # Load model and vocabulary
    model, vocab_mappings = load_model_for_generation(str(best_checkpoint), config, device)
    
    # Create sampler
    sampler = MusicSampler(config["generation"])
    
    # Generate samples
    from src.generation.sampler import generate_music_batch
    
    output_dir = Path(config["system"]["assets_dir"]) / "generated_samples"
    generated_files = generate_music_batch(
        model, sampler, vocab_mappings,
        num_samples=5, output_dir=str(output_dir), device=device
    )
    
    logger.info(f"Generated {len(generated_files)} music samples in {output_dir}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train music generation model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--model-type", type=str, choices=["lstm", "gru", "transformer"], 
                       help="Model type to train")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, help="Hidden size")
    parser.add_argument("--num-layers", type=int, help="Number of layers")
    parser.add_argument("--generate-only", action="store_true", help="Only generate samples")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate model")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.data_dir:
        config["data"]["raw_data_dir"] = args.data_dir
        config["data"]["processed_data_dir"] = str(Path(args.data_dir).parent / "processed")
    
    if args.model_type:
        config["model"]["model_type"] = args.model_type
    
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    if args.hidden_size:
        config["model"]["hidden_size"] = args.hidden_size
    
    if args.num_layers:
        config["model"]["num_layers"] = args.num_layers
    
    # Set random seed
    set_seed(config["system"]["seed"])
    
    logger.info("Starting music generation training pipeline")
    logger.info(f"Configuration: {config}")
    
    try:
        if args.generate_only:
            # Only generate samples
            generate_samples(None, None, config)
        elif args.eval_only:
            # Only evaluate existing model
            train_loader, val_loader, test_loader, processor = prepare_data(config)
            device = get_device()
            checkpoint_dir = Path(config["system"]["checkpoint_dir"])
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
            
            if checkpoint_files:
                best_checkpoint = min(checkpoint_files, key=lambda x: float(x.stem.split('-')[-1]))
                model, _ = load_model_for_generation(str(best_checkpoint), config, device)
                evaluate_model(model, test_loader, processor, config)
            else:
                logger.error("No checkpoint files found for evaluation")
        else:
            # Full training pipeline
            # Prepare data
            train_loader, val_loader, test_loader, processor = prepare_data(config)
            
            # Train model
            model = train_model(config, train_loader, val_loader, test_loader)
            
            # Evaluate model
            evaluate_model(model, test_loader, processor, config)
            
            # Generate samples
            generate_samples(model, processor, config)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
