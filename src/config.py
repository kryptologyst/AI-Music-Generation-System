"""Configuration management for music generation system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "lstm"  # lstm, gru, transformer
    input_size: int = 1
    hidden_size: int = 512
    num_layers: int = 3
    output_size: int = 128
    dropout: float = 0.2
    bidirectional: bool = False
    attention: bool = False
    attention_heads: int = 8


@dataclass
class DataConfig:
    """Data processing configuration."""
    sequence_length: int = 100
    batch_size: int = 32
    num_workers: int = 4
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_note_duration: float = 0.1
    max_note_duration: float = 4.0
    sample_rate: int = 22050


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # cosine, linear, step
    patience: int = 10
    min_delta: float = 0.001
    mixed_precision: bool = True
    accumulate_grad_batches: int = 1


@dataclass
class GenerationConfig:
    """Music generation configuration."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_length: int = 500
    seed_length: int = 100
    sampling_strategy: str = "nucleus"  # greedy, nucleus, top_k, temperature
    repetition_penalty: float = 1.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "perplexity", "nll", "diversity", "coherence", "rhythm_consistency"
    ])
    eval_batch_size: int = 16
    num_samples: int = 100


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "auto"  # auto, cpu, cuda, mps
    seed: int = 42
    deterministic: bool = True
    log_level: str = "INFO"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    assets_dir: str = "assets"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return OmegaConf.structured(cls(**config_dict))

    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = OmegaConf.to_yaml(self)
        with open(config_path, 'w') as f:
            f.write(config_dict)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return get_default_config()
