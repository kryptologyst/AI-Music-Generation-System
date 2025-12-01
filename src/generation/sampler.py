"""Music generation and sampling utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import music21
from music21 import note, chord, stream, duration
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MusicSampler:
    """Advanced sampling strategies for music generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temperature = config.get("temperature", 1.0)
        self.top_k = config.get("top_k", 50)
        self.top_p = config.get("top_p", 0.9)
        self.repetition_penalty = config.get("repetition_penalty", 1.1)
        self.sampling_strategy = config.get("sampling_strategy", "nucleus")
    
    def nucleus_sampling(self, logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        """Nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return F.softmax(logits, dim=-1)
    
    def top_k_sampling(self, logits: torch.Tensor, top_k: int = 50) -> torch.Tensor:
        """Top-k sampling."""
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        return F.softmax(logits, dim=-1)
    
    def temperature_sampling(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Temperature sampling."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        
        logits = logits / temperature
        return F.softmax(logits, dim=-1)
    
    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float = 1.1
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for i in range(logits.size(0)):
            for j in range(logits.size(1)):
                token_id = input_ids[i, j].item()
                if token_id in input_ids[i, :j]:
                    logits[i, j, token_id] /= penalty
        
        return logits
    
    def sample_token(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample a token using the configured strategy."""
        # Apply repetition penalty if input_ids provided
        if input_ids is not None and self.repetition_penalty != 1.0:
            logits = self.apply_repetition_penalty(logits, input_ids, self.repetition_penalty)
        
        # Apply sampling strategy
        if self.sampling_strategy == "nucleus":
            probs = self.nucleus_sampling(logits, self.top_p)
        elif self.sampling_strategy == "top_k":
            probs = self.top_k_sampling(logits, self.top_k)
        elif self.sampling_strategy == "temperature":
            probs = self.temperature_sampling(logits, self.temperature)
        elif self.sampling_strategy == "greedy":
            return torch.argmax(logits, dim=-1)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Sample from the probability distribution
        return torch.multinomial(probs, 1)


class MusicGenerator:
    """Main music generation class."""
    
    def __init__(self, model: torch.nn.Module, sampler: MusicSampler, vocab_mappings: Dict[str, Any]):
        self.model = model
        self.sampler = sampler
        self.note_to_int = vocab_mappings["note_to_int"]
        self.int_to_note = vocab_mappings["int_to_note"]
        self.vocab_size = vocab_mappings["vocab_size"]
        
        self.model.eval()
    
    def generate_sequence(
        self,
        seed_sequence: List[int],
        max_length: int = 500,
        device: torch.device = torch.device("cpu")
    ) -> List[int]:
        """Generate a sequence of music tokens."""
        self.model.to(device)
        
        # Convert seed to tensor
        current_sequence = torch.tensor(seed_sequence, dtype=torch.long, device=device).unsqueeze(0)
        generated_sequence = seed_sequence.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model prediction
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'init_hidden'):
                    # For RNN-based models
                    hidden = self.model.model.init_hidden(1, device)
                    output, _ = self.model(current_sequence, hidden)
                else:
                    # For transformer-based models
                    output = self.model(current_sequence)
                
                # Get logits for the last token
                logits = output[:, -1, :]
                
                # Sample next token
                next_token = self.sampler.sample_token(logits)
                next_token_id = next_token.item()
                
                # Add to generated sequence
                generated_sequence.append(next_token_id)
                
                # Update current sequence (sliding window)
                if current_sequence.size(1) >= len(seed_sequence):
                    current_sequence = torch.cat([
                        current_sequence[:, 1:],
                        next_token.unsqueeze(0)
                    ], dim=1)
                else:
                    current_sequence = torch.cat([
                        current_sequence,
                        next_token.unsqueeze(0)
                    ], dim=1)
        
        return generated_sequence
    
    def generate_music(
        self,
        seed_sequence: Optional[List[int]] = None,
        max_length: int = 500,
        device: torch.device = torch.device("cpu")
    ) -> List[int]:
        """Generate music with optional seed sequence."""
        if seed_sequence is None:
            # Generate random seed sequence
            seed_length = self.sampler.config.get("seed_length", 100)
            seed_sequence = np.random.randint(0, self.vocab_size, seed_length).tolist()
        
        return self.generate_sequence(seed_sequence, max_length, device)
    
    def ints_to_notes(self, int_sequence: List[int]) -> List[Any]:
        """Convert integer sequence to music21 notes/chords."""
        notes = []
        
        for int_val in int_sequence:
            if int_val in self.int_to_note:
                note_str = self.int_to_note[int_val]
                
                if '.' in note_str and '_' in note_str:
                    # Handle chords with duration
                    chord_part, duration_part = note_str.split('_')
                    chord_notes = chord_part.split('.')
                    chord_notes = [note.Note(int(n)) for n in chord_notes]
                    chord_obj = chord.Chord(chord_notes)
                    chord_obj.duration = duration.Duration(float(duration_part))
                    notes.append(chord_obj)
                elif '_' in note_str:
                    # Handle single notes with duration
                    pitch_part, duration_part = note_str.split('_')
                    note_obj = note.Note(int(pitch_part))
                    note_obj.duration = duration.Duration(float(duration_part))
                    notes.append(note_obj)
                else:
                    # Handle simple notes
                    notes.append(note.Note(note_str))
        
        return notes
    
    def save_midi(self, notes: List[Any], output_path: str) -> None:
        """Save generated notes as MIDI file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create music21 stream
        midi_stream = stream.Stream(notes)
        
        # Add instrument
        piano = instrument.Piano()
        midi_stream.insert(0, piano)
        
        # Write MIDI file
        midi_stream.write('midi', fp=str(output_path))
        logger.info(f"Saved generated music to {output_path}")
    
    def generate_and_save(
        self,
        output_path: str,
        seed_sequence: Optional[List[int]] = None,
        max_length: int = 500,
        device: torch.device = torch.device("cpu")
    ) -> str:
        """Generate music and save as MIDI file."""
        # Generate sequence
        generated_sequence = self.generate_music(seed_sequence, max_length, device)
        
        # Convert to notes
        notes = self.ints_to_notes(generated_sequence)
        
        # Save as MIDI
        self.save_midi(notes, output_path)
        
        return output_path


def load_model_for_generation(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a trained model for generation."""
    from ..models.architecture import create_model
    
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
    
    model = create_model(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load vocabulary
    vocab_path = Path(config["data"]["processed_data_dir"]) / "vocabulary.pkl"
    import pickle
    with open(vocab_path, "rb") as f:
        vocab_mappings = pickle.load(f)
    
    return model, vocab_mappings


def generate_music_batch(
    model: torch.nn.Module,
    sampler: MusicSampler,
    vocab_mappings: Dict[str, Any],
    num_samples: int = 5,
    output_dir: str = "generated_music",
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """Generate multiple music samples."""
    generator = MusicGenerator(model, sampler, vocab_mappings)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for i in range(num_samples):
        output_path = output_dir / f"generated_music_{i:03d}.mid"
        
        # Generate with different random seeds
        np.random.seed(i * 42)
        generated_file = generator.generate_and_save(
            str(output_path),
            max_length=500,
            device=device
        )
        
        generated_files.append(generated_file)
        logger.info(f"Generated sample {i+1}/{num_samples}: {generated_file}")
    
    return generated_files
