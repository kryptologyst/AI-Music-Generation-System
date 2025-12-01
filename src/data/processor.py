"""Data processing pipeline for music generation."""

import os
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import music21
from music21 import converter, instrument, note, chord, stream, duration
import librosa
import soundfile as sf
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class MusicDataset(Dataset):
    """Dataset class for music sequences."""
    
    def __init__(
        self,
        sequences: List[List[int]],
        sequence_length: int = 100,
        vocab_size: int = 128,
        transform: Optional[Any] = None
    ):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.transform = transform
        
        # Create input-target pairs
        self.data_pairs = []
        for seq in sequences:
            if len(seq) > sequence_length:
                for i in range(len(seq) - sequence_length):
                    input_seq = seq[i:i + sequence_length]
                    target = seq[i + sequence_length]
                    self.data_pairs.append((input_seq, target))
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target = self.data_pairs[idx]
        
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return input_tensor, target_tensor


class MusicDataProcessor:
    """Main data processing class for music files."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.note_to_int: Dict[str, int] = {}
        self.int_to_note: Dict[int, str] = {}
        self.vocab_size = 0
        
    def process_midi_file(self, midi_path: str) -> List[str]:
        """Process a single MIDI file and extract note sequences."""
        try:
            # Parse MIDI file
            midi_stream = converter.parse(midi_path)
            
            # Extract notes and chords
            notes = []
            for element in midi_stream.flat.notes:
                if isinstance(element, note.Note):
                    # Include pitch and duration information
                    note_str = f"{element.pitch.midi}_{element.duration.quarterLength}"
                    notes.append(note_str)
                elif isinstance(element, chord.Chord):
                    # Handle chords
                    chord_str = ".".join([str(n.pitch.midi) for n in element.pitches])
                    chord_str += f"_{element.duration.quarterLength}"
                    notes.append(chord_str)
            
            return notes
            
        except Exception as e:
            logger.warning(f"Error processing {midi_path}: {e}")
            return []
    
    def process_directory(self, data_dir: str) -> List[List[str]]:
        """Process all MIDI files in a directory."""
        data_dir = Path(data_dir)
        all_sequences = []
        
        midi_files = list(data_dir.glob("*.mid")) + list(data_dir.glob("*.midi"))
        
        logger.info(f"Found {len(midi_files)} MIDI files")
        
        for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
            sequence = self.process_midi_file(str(midi_file))
            if sequence:
                all_sequences.append(sequence)
        
        return all_sequences
    
    def create_vocabulary(self, all_sequences: List[List[str]]) -> None:
        """Create vocabulary mapping from all sequences."""
        all_notes = []
        for sequence in all_sequences:
            all_notes.extend(sequence)
        
        unique_notes = sorted(set(all_notes))
        self.note_to_int = {note: idx for idx, note in enumerate(unique_notes)}
        self.int_to_note = {idx: note for note, idx in self.note_to_int.items()}
        self.vocab_size = len(unique_notes)
        
        logger.info(f"Created vocabulary with {self.vocab_size} unique notes/chords")
    
    def sequences_to_ints(self, sequences: List[List[str]]) -> List[List[int]]:
        """Convert note sequences to integer sequences."""
        int_sequences = []
        for sequence in sequences:
            int_seq = [self.note_to_int[note] for note in sequence if note in self.note_to_int]
            if int_seq:
                int_sequences.append(int_seq)
        return int_sequences
    
    def create_splits(
        self,
        sequences: List[List[int]],
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Split sequences into train/val/test sets."""
        random.shuffle(sequences)
        
        n_total = len(sequences)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train + n_val]
        test_sequences = sequences[n_train + n_val:]
        
        logger.info(f"Split: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")
        
        return train_sequences, val_sequences, test_sequences
    
    def save_processed_data(
        self,
        train_sequences: List[List[int]],
        val_sequences: List[List[int]],
        test_sequences: List[List[int]],
        output_dir: str
    ) -> None:
        """Save processed data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sequences
        with open(output_dir / "train_sequences.pkl", "wb") as f:
            pickle.dump(train_sequences, f)
        
        with open(output_dir / "val_sequences.pkl", "wb") as f:
            pickle.dump(val_sequences, f)
        
        with open(output_dir / "test_sequences.pkl", "wb") as f:
            pickle.dump(test_sequences, f)
        
        # Save vocabulary
        with open(output_dir / "vocabulary.pkl", "wb") as f:
            pickle.dump({
                "note_to_int": self.note_to_int,
                "int_to_note": self.int_to_note,
                "vocab_size": self.vocab_size
            }, f)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    def load_processed_data(self, data_dir: str) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Load previously processed data."""
        data_dir = Path(data_dir)
        
        with open(data_dir / "train_sequences.pkl", "rb") as f:
            train_sequences = pickle.load(f)
        
        with open(data_dir / "val_sequences.pkl", "rb") as f:
            val_sequences = pickle.load(f)
        
        with open(data_dir / "test_sequences.pkl", "rb") as f:
            test_sequences = pickle.load(f)
        
        with open(data_dir / "vocabulary.pkl", "rb") as f:
            vocab_data = pickle.load(f)
            self.note_to_int = vocab_data["note_to_int"]
            self.int_to_note = vocab_data["int_to_note"]
            self.vocab_size = vocab_data["vocab_size"]
        
        logger.info(f"Loaded processed data from {data_dir}")
        return train_sequences, val_sequences, test_sequences


def create_data_loaders(
    train_sequences: List[List[int]],
    val_sequences: List[List[int]],
    test_sequences: List[List[int]],
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train/val/test sets."""
    
    train_dataset = MusicDataset(
        train_sequences,
        sequence_length=config["sequence_length"],
        vocab_size=config.get("vocab_size", 128)
    )
    
    val_dataset = MusicDataset(
        val_sequences,
        sequence_length=config["sequence_length"],
        vocab_size=config.get("vocab_size", 128)
    )
    
    test_dataset = MusicDataset(
        test_sequences,
        sequence_length=config["sequence_length"],
        vocab_size=config.get("vocab_size", 128)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def generate_toy_dataset(output_dir: str, num_files: int = 10) -> None:
    """Generate a toy MIDI dataset for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {num_files} toy MIDI files")
    
    for i in range(num_files):
        # Create a simple melody
        melody = stream.Stream()
        
        # Add some notes
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
        durations = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0]
        
        for pitch, dur in zip(pitches, durations):
            n = note.Note(pitch)
            n.duration = duration.Duration(dur)
            melody.append(n)
        
        # Save as MIDI
        output_path = output_dir / f"toy_melody_{i:03d}.mid"
        melody.write('midi', fp=str(output_path))
    
    logger.info(f"Generated toy dataset in {output_dir}")
