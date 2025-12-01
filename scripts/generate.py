#!/usr/bin/env python3
"""Sample generation script for music generation system."""

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
from src.generation.sampler import MusicSampler, MusicGenerator, load_model_for_generation, generate_music_batch
from src.training.trainer import get_device

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_music_samples(
    config_path: str = None,
    checkpoint_path: str = None,
    output_dir: str = "generated_music",
    num_samples: int = 5,
    max_length: int = 500,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    sampling_strategy: str = "nucleus",
    seed: int = 42
):
    """Generate music samples using a trained model."""
    
    # Setup logging
    setup_logging()
    
    # Set random seed
    set_seed(seed)
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = get_default_config()
    
    # Override generation config with command line arguments
    config["generation"]["temperature"] = temperature
    config["generation"]["top_k"] = top_k
    config["generation"]["top_p"] = top_p
    config["generation"]["sampling_strategy"] = sampling_strategy
    config["generation"]["max_length"] = max_length
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Find checkpoint if not provided
    if not checkpoint_path:
        checkpoint_dir = Path(config["system"]["checkpoint_dir"])
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        
        if not checkpoint_files:
            logger.error("No checkpoint files found. Please train a model first or specify --checkpoint-path")
            return
        
        # Use the best checkpoint
        checkpoint_path = str(min(checkpoint_files, key=lambda x: float(x.stem.split('-')[-1])))
        logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Load model and vocabulary
    logger.info("Loading model and vocabulary...")
    model, vocab_mappings = load_model_for_generation(checkpoint_path, config, device)
    
    # Create sampler
    sampler = MusicSampler(config["generation"])
    
    # Create generator
    generator = MusicGenerator(model, sampler, vocab_mappings)
    
    # Generate samples
    logger.info(f"Generating {num_samples} music samples...")
    generated_files = generate_music_batch(
        model, sampler, vocab_mappings,
        num_samples=num_samples,
        output_dir=output_dir,
        device=device
    )
    
    logger.info(f"Generated {len(generated_files)} music samples:")
    for i, file_path in enumerate(generated_files, 1):
        logger.info(f"  {i}. {file_path}")
    
    return generated_files


def interactive_generation(config_path: str = None, checkpoint_path: str = None):
    """Interactive music generation with custom parameters."""
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = get_default_config()
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Find checkpoint if not provided
    if not checkpoint_path:
        checkpoint_dir = Path(config["system"]["checkpoint_dir"])
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        
        if not checkpoint_files:
            logger.error("No checkpoint files found. Please train a model first or specify --checkpoint-path")
            return
        
        checkpoint_path = str(min(checkpoint_files, key=lambda x: float(x.stem.split('-')[-1])))
    
    # Load model and vocabulary
    logger.info("Loading model and vocabulary...")
    model, vocab_mappings = load_model_for_generation(checkpoint_path, config, device)
    
    print("\n" + "="*60)
    print("INTERACTIVE MUSIC GENERATION")
    print("="*60)
    print("Generate music with custom parameters!")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            # Get user input
            print("\nGeneration Parameters:")
            temperature = float(input("Temperature (0.1-2.0, default 1.0): ") or "1.0")
            top_k = int(input("Top-k (1-100, default 50): ") or "50")
            top_p = float(input("Top-p (0.1-1.0, default 0.9): ") or "0.9")
            max_length = int(input("Max length (100-1000, default 500): ") or "500")
            num_samples = int(input("Number of samples (1-10, default 1): ") or "1")
            
            # Validate inputs
            temperature = max(0.1, min(2.0, temperature))
            top_k = max(1, min(100, top_k))
            top_p = max(0.1, min(1.0, top_p))
            max_length = max(100, min(1000, max_length))
            num_samples = max(1, min(10, num_samples))
            
            # Update config
            config["generation"]["temperature"] = temperature
            config["generation"]["top_k"] = top_k
            config["generation"]["top_p"] = top_p
            config["generation"]["max_length"] = max_length
            
            # Create sampler and generator
            sampler = MusicSampler(config["generation"])
            generator = MusicGenerator(model, sampler, vocab_mappings)
            
            # Generate samples
            print(f"\nGenerating {num_samples} sample(s) with:")
            print(f"  Temperature: {temperature}")
            print(f"  Top-k: {top_k}")
            print(f"  Top-p: {top_p}")
            print(f"  Max length: {max_length}")
            
            output_dir = Path("interactive_generated")
            output_dir.mkdir(exist_ok=True)
            
            generated_files = generate_music_batch(
                model, sampler, vocab_mappings,
                num_samples=num_samples,
                output_dir=str(output_dir),
                device=device
            )
            
            print(f"\nGenerated {len(generated_files)} music samples:")
            for i, file_path in enumerate(generated_files, 1):
                print(f"  {i}. {file_path}")
            
            # Ask if user wants to continue
            continue_gen = input("\nGenerate more samples? (y/n): ").lower()
            if continue_gen not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error during generation: {e}")
    
    print("\nThank you for using the music generation system!")


def main():
    """Main function for sample generation."""
    parser = argparse.ArgumentParser(description="Generate music samples")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="generated_music", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum length of generated sequences")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--sampling-strategy", type=str, default="nucleus", 
                       choices=["greedy", "nucleus", "top_k", "temperature"],
                       help="Sampling strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_generation(args.config, args.checkpoint_path)
    else:
        generate_music_samples(
            config_path=args.config,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sampling_strategy=args.sampling_strategy,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
