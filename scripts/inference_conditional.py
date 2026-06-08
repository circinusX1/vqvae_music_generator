import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
from src.train_vqvae import VQVAEModel
from src.models.generator import MusicTransformer
from src.utils.audio_processing import setup_device
# Corrected import location for Scaled Dot Product Attention context backend
from torch.nn.attention import sdpa_kernel, SDPBackend



def print_vram_usage(milestone_name):
    """Helper to print current and max VRAM utilization in Megabytes"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"--- VRAM [{milestone_name}] --- Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB | Peak: {max_allocated:.2f}MB\n")

def load_and_preprocess_reference(file_path, target_sr, target_duration):
    """Loads a reference audio file and reshapes it to match training specs"""
    waveform, sr = torchaudio.load(file_path)
    
    # Mixdown stereo to mono channel
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if sample rate doesn't match config specifications
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # Crop or Pad audio precisely to the duration_sec window
    target_samples = int(target_duration * target_sr)
    if waveform.size(1) > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.size(1) < target_samples:
        pad_amount = target_samples - waveform.size(1)
        waveform = F.pad(waveform, (0, pad_amount))
        
    return waveform.unsqueeze(0) # Output shape: [1, 1, T_samples]

def generate_conditioned_music(ref_audio_path, output_path="output_conditioned.wav"):
    # Reset tracking statistics at the entry milestone
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    print_vram_usage("Script Execution Start")

    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = setup_device(cfg['training']['device'])
    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    target_duration_sec = cfg['training']['target_generation_sec'] # 15s
    ref_window_sec = cfg['dataset']['duration_sec'] # 2s (or matching your config)
    
    tokens_per_second = sample_rate / stride
    total_tokens_needed = int(target_duration_sec * tokens_per_second)
    max_context_window = int(ref_window_sec * tokens_per_second) # Sliding boundary limit
    cudamem  = torch.cuda.get_device_properties(0).total_memory;
    print("cuda memeory =", cudamem)
    print("tokens_per_second =", tokens_per_second)
    print("max_context_window =", max_context_window)
    
    print("-------------------------------------------------------------------------------")
    # Load Models
    if cudamem < 9000000000:
        max_context_window = 1024 

    print("Loading VQ-VAE model...")
    vqvae = VQVAEModel(cfg).to(device)
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae.eval()
    print_vram_usage("VQ-VAE Weights Loaded")
    
    print("Loading Prior Transformer model...")
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], cfg['generator']['num_layers'], cfg['generator']['num_heads']
    ).to(device)
    transformer.load_state_dict(torch.load(cfg['training']['generator_path'], map_location=device))
    transformer.eval()
    print_vram_usage("Transformer Weights Loaded")

    # 1. Process Reference Song and convert to Codebook Indices
    print(f"Reading and encoding reference track: {ref_audio_path}...")
    ref_waveform = load_and_preprocess_reference(ref_audio_path, sample_rate, ref_window_sec).to(device)
    
    with torch.no_grad():
        z_ref = vqvae.encoder(ref_waveform)
        _, _, ref_indices = vqvae.quantizer(z_ref) # Shape: [1, T_ref_tokens]
        
    sos_token_id = cfg['generator']['num_embeddings'] # Index 512
    
    # 2. Prime the sequence using the reference audio tokens instead of starting blank
    sos_tokens = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
    generated_sequence = torch.cat([sos_tokens, ref_indices], dim=1)
    
    initial_token_count = generated_sequence.size(1)
    tokens_to_generate = total_tokens_needed - initial_token_count + 1
    
    print(f"Primed model with {initial_token_count} tokens from reference audio.")
    print(f"Autoregressively generating remaining {tokens_to_generate} tokens...")

    # 3. Sliding window continuation generation loop
    for i in range(tokens_to_generate):
        if generated_sequence.size(1) > max_context_window:
            context = generated_sequence[:, -max_context_window:]
        else:
            context = generated_sequence
            
        with torch.no_grad():
            import torch.nn.attention as att
            
            allowed_backends = []
            for name in dir(att.SDPBackend):
                # Match either 'FLASH' or 'flash_attention' while ignoring the slow 'math' layer
                if ('flash' in name.lower() or 'efficient' in name.lower()) and not name.startswith('__'):
                    allowed_backends.append(getattr(att.SDPBackend, name))
            
            # Fallback protection: if dynamic parsing yields nothing, default to all available
            if not allowed_backends:
                print("Warning: Could not dynamically resolve accelerated backends. Using system defaults.")
                logits = transformer(context)
            else:
                # Pass the resolved backend list as positional arguments into the modern context manager
                with att.sdpa_kernel(allowed_backends):
                    logits = transformer(context)            
            
            next_token_logits = logits[:, -1, :]
            
            # Apply Temperature sampling to balance style fidelity and variation
            temperature = cfg["generator"]["sampling_temp"]
            
            filtered_logits = next_token_logits / temperature
            filtered_logits[:, sos_token_id] = -float('Inf')  # Ban loop resets
            
            # 2. TOP-K FILTERING FIX
            top_k = 10  # Only look at the top 10 most confident token choices
            top_logits, top_indices = torch.topk(filtered_logits, top_k, dim=-1)
            
            # Create a mask of negative infinities for everything else
            mask = torch.full_list_like(filtered_logits, fill_value=-float('Inf')) if hasattr(torch, 'full_list_like') else torch.full(filtered_logits.shape, -float('Inf'), device=device)
            mask.scatter_(-1, top_indices, top_logits)
            
            # 3. Sample safely from the filtered options
            probabilities = torch.softmax(mask, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)            

            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{tokens_to_generate} additional tokens... (Current Token: {next_token.item()})")
            print_vram_usage(f"Generation Progress {i + 1}")

    # Strip away initialization SOS token
    final_indices = generated_sequence[:, 1:]
    print(f"Token map generation complete. Array length: {final_indices.shape[1]} entries.")
    print_vram_usage("Autoregression Loop Finished")
    
    # Drop Transformer memory footprint to make space for the decoder execution
    del transformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_vram_usage("Transformer Deleted / Cache Flushed")

    # 4. Chunked VQ-VAE Decoder Pass to mitigate CUDA Out of Memory spikes
    print("Decoding discrete token map into raw continuous audio waveform (Chunked)...")
    
    # Define decode chunk size (2 seconds worth of tokens at a time)
    decode_chunk_size = int(2 * tokens_per_second) 
    generated_audio_pieces = []
    
    # Process through tokens in small increments to safeguard VRAM
    for chunk_idx, start_idx in enumerate(range(0, final_indices.size(1), decode_chunk_size)):
        end_idx = min(start_idx + decode_chunk_size, final_indices.size(1))
        token_chunk = final_indices[:, start_idx:end_idx]
        
        with torch.no_grad():
            # Look up codebook vectors for just this chunk
            quantized_vectors = vqvae.quantizer.embedding(token_chunk)
            quantized_vectors = quantized_vectors.permute(0, 2, 1).contiguous()
            
            # Decode chunk back to raw waveform slice
            audio_slice = vqvae.decoder(quantized_vectors)
            generated_audio_pieces.append(audio_slice.cpu()) # Move to system RAM immediately
            
        if chunk_idx % 2 == 0:
            print_vram_usage(f"Decoding Step {chunk_idx}")
            
    # Concatenate all raw audio pieces along the time axis
    generated_waveform = torch.cat(generated_audio_pieces, dim=-1)
    
    print_vram_usage("Decoder Steps Completed")
    
    # Export back into raw storage directory
    torchaudio.save(output_path, generated_waveform.squeeze(0), sample_rate)
    print(f"=== Success! Conditioned file successfully rendered to: {os.path.abspath(output_path)} ===")

if __name__ == "__main__":
    # Define the fallback file verification priority map
    possible_extensions = ["./data/raw/reference.mp3", "./data/raw/reference.wav"]
    reference_file = None
    
    # Check the base directory for the existence of either file structure
    for filename in possible_extensions:
        if os.path.exists(filename):
            reference_file = filename
            break
            
    if reference_file is not None:
        print(f"Found reference source asset: {reference_file}")
        generate_conditioned_music(reference_file)
    else:
        print(f"Error: Neither 'reference.mp3' nor 'reference.wav' was located in your target directories.")
