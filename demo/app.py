"""Streamlit demo interface for music generation system."""

import streamlit as st
import torch
import numpy as np
import random
from pathlib import Path
import sys
import logging
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config, get_default_config
from src.generation.sampler import MusicSampler, MusicGenerator, load_model_for_generation
from src.training.trainer import get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@st.cache_resource
def load_model_and_config(checkpoint_path: str, config_path: str = None):
    """Load model and configuration (cached for performance)."""
    try:
        # Load configuration
        if config_path and Path(config_path).exists():
            config = load_config(config_path)
        else:
            config = get_default_config()
        
        # Get device
        device = get_device()
        
        # Load model and vocabulary
        model, vocab_mappings = load_model_for_generation(checkpoint_path, config, device)
        
        return model, vocab_mappings, config, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Music Generation System",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 AI Music Generation System")
    st.markdown("Generate original music using deep learning models trained on MIDI data.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    checkpoint_dir = Path("checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
    
    if not checkpoint_files:
        st.error("No trained models found! Please train a model first using the training script.")
        st.stop()
    
    checkpoint_options = {str(f): f.name for f in checkpoint_files}
    selected_checkpoint = st.sidebar.selectbox(
        "Select Model Checkpoint",
        options=list(checkpoint_options.keys()),
        format_func=lambda x: checkpoint_options[x]
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, vocab_mappings, config, device = load_model_and_config(selected_checkpoint)
    
    if model is None:
        st.error("Failed to load model!")
        st.stop()
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    # Sampling strategy
    sampling_strategy = st.sidebar.selectbox(
        "Sampling Strategy",
        ["nucleus", "top_k", "temperature", "greedy"],
        help="Different sampling strategies for music generation"
    )
    
    # Temperature
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher values make generation more random"
    )
    
    # Top-k
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        help="Number of top tokens to consider"
    )
    
    # Top-p
    top_p = st.sidebar.slider(
        "Top-p (Nucleus)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Cumulative probability threshold for nucleus sampling"
    )
    
    # Max length
    max_length = st.sidebar.slider(
        "Max Length",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Maximum length of generated sequence"
    )
    
    # Number of samples
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of music pieces to generate"
    )
    
    # Random seed
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        help="Seed for reproducible generation"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Generate Music")
        
        # Generate button
        if st.button("🎼 Generate Music", type="primary", use_container_width=True):
            # Set seed
            set_seed(seed)
            
            # Update config
            generation_config = {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "sampling_strategy": sampling_strategy,
                "max_length": max_length
            }
            
            # Create sampler and generator
            sampler = MusicSampler(generation_config)
            generator = MusicGenerator(model, sampler, vocab_mappings)
            
            # Generate samples
            with st.spinner("Generating music..."):
                generated_files = []
                
                for i in range(num_samples):
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_file:
                        output_path = tmp_file.name
                    
                    # Generate music
                    generator.generate_and_save(
                        output_path,
                        max_length=max_length,
                        device=device
                    )
                    
                    generated_files.append(output_path)
            
            # Display results
            st.success(f"Generated {num_samples} music piece(s)!")
            
            # Show download links
            for i, file_path in enumerate(generated_files, 1):
                with open(file_path, "rb") as file:
                    st.download_button(
                        label=f"Download Sample {i}",
                        data=file.read(),
                        file_name=f"generated_music_{i}.mid",
                        mime="audio/midi"
                    )
            
            # Clean up temporary files
            for file_path in generated_files:
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    with col2:
        st.header("Model Information")
        
        # Display model info
        st.info(f"""
        **Model Type:** {config['model']['model_type'].upper()}
        
        **Hidden Size:** {config['model']['hidden_size']}
        
        **Number of Layers:** {config['model']['num_layers']}
        
        **Vocabulary Size:** {vocab_mappings['vocab_size']}
        
        **Device:** {device}
        """)
        
        # Sampling strategy explanation
        st.header("Sampling Strategies")
        
        strategy_descriptions = {
            "nucleus": "**Nucleus (Top-p):** Samples from tokens whose cumulative probability is below the threshold. Good balance of creativity and coherence.",
            "top_k": "**Top-k:** Samples only from the k most likely tokens. More conservative generation.",
            "temperature": "**Temperature:** Applies temperature scaling to logits. Higher temperature = more random.",
            "greedy": "**Greedy:** Always picks the most likely token. Most conservative but potentially repetitive."
        }
        
        st.markdown(strategy_descriptions[sampling_strategy])
        
        # Tips
        st.header("Tips")
        st.markdown("""
        - **Temperature 0.5-0.8:** More conservative, structured music
        - **Temperature 1.0-1.5:** Balanced creativity and coherence  
        - **Temperature 1.5-2.0:** More experimental, creative music
        - **Lower Top-p:** More focused generation
        - **Higher Top-p:** More diverse generation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎵 AI Music Generation System | Built with PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
