"""Training system for music generation models using PyTorch Lightning."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
from torchmetrics import MeanMetric
import wandb

from ..models.architecture import create_model
from ..data.processor import MusicDataset

logger = logging.getLogger(__name__)


class MusicGenerationModule(pl.LightningModule):
    """PyTorch Lightning module for music generation training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model
        model_config = {
            "model_type": config["model"]["model_type"],
            "input_size": config["model"]["input_size"],
            "hidden_size": config["model"]["hidden_size"],
            "num_layers": config["model"]["num_layers"],
            "output_size": config["model"]["output_size"],
            "dropout": config["model"]["dropout"],
            "bidirectional": config["model"]["bidirectional"],
            "attention": config["model"]["attention"],
            "attention_heads": config["model"]["attention_heads"]
        }
        
        self.model = create_model(model_config)
        self.vocab_size = config["model"]["output_size"]
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_perplexity = MeanMetric()
        self.val_perplexity = MeanMetric()
        
        logger.info(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, hidden)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_seq, target = batch
        
        # Forward pass
        output, _ = self.model(input_seq)
        
        # Reshape for loss computation
        output = output.view(-1, self.vocab_size)
        target = target.view(-1)
        
        # Compute loss
        loss = self.criterion(output, target)
        
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.train_loss(loss)
        self.train_perplexity(perplexity)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_seq, target = batch
        
        # Forward pass
        output, _ = self.model(input_seq)
        
        # Reshape for loss computation
        output = output.view(-1, self.vocab_size)
        target = target.view(-1)
        
        # Compute loss
        loss = self.criterion(output, target)
        
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.val_loss(loss)
        self.val_perplexity(perplexity)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", perplexity, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        if self.config["training"].get("optimizer", "adam") == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["weight_decay"]
            )
        else:
            optimizer = Adam(
                self.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["weight_decay"]
            )
        
        # Scheduler
        scheduler_config = self.config["training"].get("scheduler", "cosine")
        
        if scheduler_config == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config["training"]["num_epochs"],
                eta_min=self.config["training"]["learning_rate"] * 0.01
            )
        elif scheduler_config == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.config["training"]["num_epochs"] // 3,
                gamma=0.1
            )
        elif scheduler_config == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.config["training"]["num_epochs"]
            )
        else:
            scheduler = None
        
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        avg_loss = self.train_loss.compute()
        avg_perplexity = self.train_perplexity.compute()
        
        logger.info(f"Epoch {self.current_epoch}: Train Loss = {avg_loss:.4f}, Train Perplexity = {avg_perplexity:.4f}")
        
        # Reset metrics
        self.train_loss.reset()
        self.train_perplexity.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        avg_loss = self.val_loss.compute()
        avg_perplexity = self.val_perplexity.compute()
        
        logger.info(f"Epoch {self.current_epoch}: Val Loss = {avg_loss:.4f}, Val Perplexity = {avg_perplexity:.4f}")
        
        # Reset metrics
        self.val_loss.reset()
        self.val_perplexity.reset()


class MusicTrainer:
    """Main trainer class for music generation models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.setup_directories()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config["system"]["log_level"]
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_directories(self):
        """Create necessary directories."""
        Path(self.config["system"]["log_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["system"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["system"]["assets_dir"]).mkdir(parents=True, exist_ok=True)
    
    def setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config["system"]["checkpoint_dir"],
            filename="music-model-{epoch:02d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            min_delta=self.config["training"]["min_delta"],
            patience=self.config["training"]["patience"],
            mode="min",
            verbose=True
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def setup_logger(self):
        """Setup logging system."""
        loggers = []
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=self.config["system"]["log_dir"],
            name="music_generation",
            version=None
        )
        loggers.append(tb_logger)
        
        # Weights & Biases logger (optional)
        if self.config.get("wandb", {}).get("enabled", False):
            wandb_logger = WandbLogger(
                project=self.config["wandb"]["project"],
                name=f"music_generation_{self.config['model']['model_type']}",
                config=self.config
            )
            loggers.append(wandb_logger)
        
        return loggers
    
    def train(
        self,
        train_loader,
        val_loader,
        test_loader=None
    ) -> pl.LightningModule:
        """Train the music generation model."""
        
        # Set random seeds for reproducibility
        pl.seed_everything(self.config["system"]["seed"])
        
        # Create model
        model = MusicGenerationModule(self.config)
        
        # Setup callbacks and loggers
        callbacks = self.setup_callbacks()
        loggers = self.setup_logger()
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["num_epochs"],
            callbacks=callbacks,
            logger=loggers,
            gradient_clip_val=self.config["training"]["gradient_clip_val"],
            accumulate_grad_batches=self.config["training"]["accumulate_grad_batches"],
            precision=16 if self.config["training"]["mixed_precision"] else 32,
            deterministic=self.config["system"]["deterministic"],
            devices=1,
            accelerator="auto"
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Test the model if test loader is provided
        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            trainer.test(model, test_loader)
        
        logger.info("Training completed!")
        
        return model
    
    def resume_training(
        self,
        checkpoint_path: str,
        train_loader,
        val_loader,
        test_loader=None
    ) -> pl.LightningModule:
        """Resume training from a checkpoint."""
        
        # Create model
        model = MusicGenerationModule(self.config)
        
        # Setup callbacks and loggers
        callbacks = self.setup_callbacks()
        loggers = self.setup_logger()
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["num_epochs"],
            callbacks=callbacks,
            logger=loggers,
            gradient_clip_val=self.config["training"]["gradient_clip_val"],
            accumulate_grad_batches=self.config["training"]["accumulate_grad_batches"],
            precision=16 if self.config["training"]["mixed_precision"] else 32,
            deterministic=self.config["system"]["deterministic"],
            devices=1,
            accelerator="auto"
        )
        
        # Resume training
        logger.info(f"Resuming training from {checkpoint_path}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
        
        # Test the model if test loader is provided
        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            trainer.test(model, test_loader)
        
        logger.info("Training completed!")
        
        return model


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
