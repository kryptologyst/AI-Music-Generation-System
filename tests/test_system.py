"""Unit tests for the music generation system."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.architecture import MusicLSTM, MusicGRU, MusicTransformer, create_model
from src.data.processor import MusicDataset, MusicDataProcessor
from src.generation.sampler import MusicSampler
from src.evaluation.metrics import MusicMetrics
from src.utils import set_deterministic, get_device, count_parameters


class TestModelArchitecture:
    """Test model architecture components."""
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation and forward pass."""
        model = MusicLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=128,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size, seq_len = 4, 50
        x = torch.randint(0, 128, (batch_size, seq_len))
        
        output, hidden = model(x)
        
        assert output.shape == (batch_size, seq_len, 128)
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2
    
    def test_gru_model_creation(self):
        """Test GRU model creation and forward pass."""
        model = MusicGRU(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=128,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size, seq_len = 4, 50
        x = torch.randint(0, 128, (batch_size, seq_len))
        
        output, hidden = model(x)
        
        assert output.shape == (batch_size, seq_len, 128)
        assert isinstance(hidden, torch.Tensor)
    
    def test_transformer_model_creation(self):
        """Test Transformer model creation and forward pass."""
        model = MusicTransformer(
            vocab_size=128,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size, seq_len = 4, 50
        x = torch.randint(0, 128, (batch_size, seq_len))
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_model_factory(self):
        """Test model factory function."""
        configs = [
            {"model_type": "lstm", "input_size": 1, "hidden_size": 64, "num_layers": 2, "output_size": 128},
            {"model_type": "gru", "input_size": 1, "hidden_size": 64, "num_layers": 2, "output_size": 128},
            {"model_type": "transformer", "vocab_size": 128, "d_model": 64, "nhead": 4, "num_layers": 2}
        ]
        
        for config in configs:
            model = create_model(config)
            assert model is not None
            assert isinstance(model, torch.nn.Module)


class TestDataProcessing:
    """Test data processing components."""
    
    def test_music_dataset(self):
        """Test MusicDataset class."""
        sequences = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        ]
        
        dataset = MusicDataset(sequences, sequence_length=5, vocab_size=128)
        
        assert len(dataset) > 0
        
        # Test getting an item
        input_seq, target = dataset[0]
        assert input_seq.shape == (5,)
        assert isinstance(target, torch.Tensor)
    
    def test_music_data_processor(self):
        """Test MusicDataProcessor class."""
        processor = MusicDataProcessor({})
        
        # Test vocabulary creation
        sequences = [["C4_1.0", "D4_0.5", "E4_1.0"], ["F4_1.0", "G4_0.5", "A4_1.0"]]
        processor.create_vocabulary(sequences)
        
        assert processor.vocab_size > 0
        assert len(processor.note_to_int) > 0
        assert len(processor.int_to_note) > 0
        
        # Test sequence conversion
        int_sequences = processor.sequences_to_ints(sequences)
        assert len(int_sequences) == len(sequences)
        assert all(isinstance(seq, list) for seq in int_sequences)


class TestSampling:
    """Test sampling components."""
    
    def test_music_sampler(self):
        """Test MusicSampler class."""
        config = {
            "temperature": 1.0,
            "top_k": 10,
            "top_p": 0.9,
            "sampling_strategy": "nucleus"
        }
        
        sampler = MusicSampler(config)
        
        # Test sampling
        logits = torch.randn(1, 128)
        probs = sampler.nucleus_sampling(logits, 0.9)
        
        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_sampling_strategies(self):
        """Test different sampling strategies."""
        config = {"temperature": 1.0, "top_k": 10, "top_p": 0.9}
        sampler = MusicSampler(config)
        
        logits = torch.randn(1, 128)
        
        # Test different strategies
        strategies = ["nucleus", "top_k", "temperature"]
        for strategy in strategies:
            sampler.sampling_strategy = strategy
            token = sampler.sample_token(logits)
            assert isinstance(token, torch.Tensor)
            assert token.shape == (1, 1)


class TestEvaluation:
    """Test evaluation components."""
    
    def test_music_metrics(self):
        """Test MusicMetrics class."""
        metrics = MusicMetrics(vocab_size=128)
        
        # Test perplexity calculation
        logits = torch.randn(10, 128)
        targets = torch.randint(0, 128, (10,))
        
        perplexity = metrics.perplexity(logits, targets)
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_diversity_metrics(self):
        """Test diversity metrics calculation."""
        metrics = MusicMetrics(vocab_size=128)
        
        sequences = [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ]
        
        diversity = metrics.diversity_metrics(sequences)
        
        assert "distinct_1" in diversity
        assert "distinct_2" in diversity
        assert "self_bleu" in diversity
        
        assert 0 <= diversity["distinct_1"] <= 1
        assert 0 <= diversity["distinct_2"] <= 1


class TestUtilities:
    """Test utility functions."""
    
    def test_deterministic_seeding(self):
        """Test deterministic seeding."""
        set_deterministic(42)
        
        # Generate some random numbers
        rand1 = torch.randn(5)
        rand2 = np.random.randn(5)
        
        # Reset seed and generate again
        set_deterministic(42)
        rand3 = torch.randn(5)
        rand4 = np.random.randn(5)
        
        # Should be the same
        assert torch.allclose(rand1, rand3)
        assert np.allclose(rand2, rand4)
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = MusicLSTM(1, 64, 2, 128)
        param_count = count_parameters(model)
        assert param_count > 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_generation(self):
        """Test end-to-end music generation pipeline."""
        # Create a simple model
        model = MusicLSTM(1, 32, 1, 64)
        
        # Create sampler
        sampler = MusicSampler({"temperature": 1.0, "top_k": 10, "top_p": 0.9})
        
        # Create vocabulary mappings
        vocab_mappings = {
            "note_to_int": {f"note_{i}": i for i in range(64)},
            "int_to_note": {i: f"note_{i}" for i in range(64)},
            "vocab_size": 64
        }
        
        # Test generation
        from src.generation.sampler import MusicGenerator
        
        generator = MusicGenerator(model, sampler, vocab_mappings)
        
        # Generate a short sequence
        seed_sequence = [0, 1, 2, 3, 4]
        generated = generator.generate_sequence(seed_sequence, max_length=10)
        
        assert len(generated) == 10
        assert all(isinstance(x, int) for x in generated)


if __name__ == "__main__":
    pytest.main([__file__])
