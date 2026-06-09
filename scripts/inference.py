
import yaml
import torch
import torchaudio
import torch.nn as nn
import os
import time
from src.train_vqvae import VQVAEModel
from src.models.generator import MusicTransformer
from src.utils.audio_processing import setup_device

@torch.no_grad()
def generate_long_music(output_path="output_15s.wav", target_duration_sec=15):
    # Load configuration
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = setup_device(cfg['training']['device'])
    
    # 1. Architecture Parameters
    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    tokens_per_second = sample_rate / stride
    total_tokens_needed = int(target_duration_sec * tokens_per_second)
    max_context_window = int(cfg['dataset']['duration_sec'] * tokens_per_second)
    
    # 2. Load VQ-VAE
    vqvae = VQVAEModel(cfg).to(device)
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae.eval()
    
    # 3. Load Transformer
    # Loading without 'style_idx' logic as requested
    checkpoint = torch.load(cfg['training']['generator_path'], map_location=device)
    
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], 
        cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], 
        cfg['generator']['num_layers'], 
        cfg['generator']['num_heads']
    )
    
    transformer.load_state_dict(checkpoint, strict=False)
    transformer.to(device).eval()
    
    # 4. Generation Setup
    # Start with strictly SOS token (512)
    sos_token_id = cfg['generator']['num_embeddings']
    generated_sequence = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
    
    print(f"Generating {total_tokens_needed} tokens...")
    start_time = time.time()
    
    # 5. Autoregressive Loop
    for i in range(total_tokens_needed):
        # Sliding context window
        context = generated_sequence[:, -max_context_window:]
            
        # Forward pass (no style_idx)
        logits = transformer(context)
        
        # Sampling
        temp = cfg['generator'].get('sampling_temp', 0.45)
        next_token_logits = logits[:, -1, :] / temp
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"Tokens: {i+1}/{total_tokens_needed} | Speed: {(i+1)/elapsed:.1f} tok/s")
            
    # 6. Chunked VQ-VAE Decoding
    print("Decoding discrete tokens to waveform...")
    final_indices = generated_sequence[:, 1:] # Strip SOS token only
    chunk_size = 2048
    audio_pieces = []
    
    for start in range(0, final_indices.size(1), chunk_size):
        chunk = final_indices[:, start : start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_pieces.append(vqvae.decoder(z_q).cpu())
        
    torchaudio.save(output_path, torch.cat(audio_pieces, dim=-1).squeeze(0), sample_rate)
    print(f"=== Success! Audio rendered to: {os.path.abspath(output_path)} ===")

if __name__ == "__main__":
    generate_long_music()

    