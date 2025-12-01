# AI Music Generation System

A production-ready music generation system using deep learning models trained on MIDI data. This project implements LSTM, GRU, and Transformer architectures with advanced sampling strategies and comprehensive evaluation metrics.

## Features

- **Multiple Model Architectures**: LSTM, GRU, and Transformer-based music generation
- **Advanced Sampling**: Nucleus, top-k, temperature, and greedy sampling strategies
- **Comprehensive Evaluation**: Music-specific metrics including rhythm consistency and coherence
- **Interactive Demo**: Streamlit-based web interface for easy music generation
- **Production Ready**: Proper logging, checkpointing, and configuration management
- **Device Support**: Automatic CUDA/MPS/CPU device detection and usage

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/AI-Music-Generation-System.git
cd AI-Music-Generation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train a model:
```bash
python scripts/train.py --model-type lstm --epochs 50
```

4. Generate music:
```bash
python scripts/generate.py --num-samples 5
```

5. Launch interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
music-generation-system/
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── models/            # Model architectures
│   ├── data/              # Data processing
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics
│   └── generation/        # Music generation
├── configs/               # Configuration files
├── scripts/               # Training and generation scripts
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Generated samples and reports
├── data/                  # Data directory
│   ├── raw/              # Raw MIDI files
│   └── processed/        # Processed data
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```

## Usage

### Training

Train a model with default configuration:
```bash
python scripts/train.py
```

Train with custom parameters:
```bash
python scripts/train.py \
    --model-type transformer \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.0005 \
    --hidden-size 512
```

Train with custom config file:
```bash
python scripts/train.py --config configs/transformer.yaml
```

### Generation

Generate music samples:
```bash
python scripts/generate.py \
    --num-samples 5 \
    --temperature 1.2 \
    --max-length 500
```

Interactive generation:
```bash
python scripts/generate.py --interactive
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/train.py --eval-only
```

## Configuration

The system uses YAML configuration files. Key configuration options:

### Model Configuration
- `model_type`: Architecture type (lstm, gru, transformer)
- `hidden_size`: Hidden layer size
- `num_layers`: Number of layers
- `dropout`: Dropout rate
- `attention`: Enable attention mechanism

### Training Configuration
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `scheduler`: Learning rate scheduler
- `mixed_precision`: Enable mixed precision training

### Generation Configuration
- `temperature`: Sampling temperature
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter
- `sampling_strategy`: Sampling method
- `max_length`: Maximum generation length

## Model Architectures

### LSTM Model
- Bidirectional LSTM with optional attention
- Good for sequential music generation
- Configurable hidden size and layers

### GRU Model
- Gated Recurrent Unit architecture
- Faster training than LSTM
- Similar performance characteristics

### Transformer Model
- Self-attention based architecture
- Better long-range dependencies
- Requires more computational resources

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Perplexity**: Model's uncertainty in predictions
- **Negative Log-Likelihood**: Log-likelihood of test data
- **Diversity Metrics**: Distinct-1, Distinct-2, Self-BLEU
- **Rhythm Consistency**: Consistency of rhythmic patterns
- **Coherence**: Musical interval consistency

## Sampling Strategies

### Nucleus Sampling (Top-p)
Samples from tokens whose cumulative probability is below the threshold. Provides good balance between creativity and coherence.

### Top-k Sampling
Samples only from the k most likely tokens. More conservative generation.

### Temperature Sampling
Applies temperature scaling to logits. Higher temperature increases randomness.

### Greedy Sampling
Always selects the most likely token. Most conservative but potentially repetitive.

## Interactive Demo

The Streamlit demo provides an intuitive interface for music generation:

1. Select a trained model checkpoint
2. Adjust generation parameters
3. Generate and download music samples
4. Experiment with different sampling strategies

Launch the demo:
```bash
streamlit run demo/app.py
```

## Data Format

The system processes MIDI files and extracts:
- Note pitches (MIDI note numbers)
- Note durations (quarter note lengths)
- Chord information

Supported formats:
- `.mid` files
- `.midi` files

## Model Cards

### LSTM Model
- **Architecture**: Bidirectional LSTM with attention
- **Parameters**: ~2M parameters (512 hidden size, 3 layers)
- **Training Time**: ~2 hours on GPU
- **Performance**: Good for short to medium sequences

### Transformer Model
- **Architecture**: Multi-head self-attention
- **Parameters**: ~5M parameters (512 hidden size, 6 layers)
- **Training Time**: ~4 hours on GPU
- **Performance**: Excellent for long sequences

## Limitations

- Generated music may not follow traditional music theory rules
- Quality depends heavily on training data
- No explicit control over musical style or genre
- Generated sequences may contain repetitive patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch and PyTorch Lightning
- Uses music21 for MIDI processing
- Streamlit for the demo interface
- Inspired by various music generation research

## Citation

If you use this code in your research, please cite:

```bibtex
@software{music_generation_system,
  title={AI Music Generation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/AI-Music-Generation-System}
}
```
# AI-Music-Generation-System
