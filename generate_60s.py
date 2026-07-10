import yaml
import torch
import torchaudio
import os
from tqdm import tqdm

# Latest project imports
from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device

@torch.no_grad()
def generate_60s(output_path="output_60s.wav", 
                 target_duration_sec=60, 
                 temperature=0.85, 
                 top_k=40):
    
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")

    # ================== Parameters ==================
    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    total_tokens = int(target_duration_sec * sample_rate / stride)
    max_context = 256

    print(f"🎵 Generating {target_duration_sec}s music → ~{total_tokens:,} tokens")
    print(f"Sampling settings → Temperature: {temperature} | Top-K: {top_k}")

    # ================== Load Models ==================
    print("Loading VQ-VAE...")
    vqvae = VQVAEModel(cfg).to(device).eval()
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))

    print("Loading Transformer...")
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'],
        cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'],
        cfg['generator']['num_layers'],
        cfg['generator']['num_heads']
    ).to(device).eval()
    transformer.load_state_dict(torch.load(cfg['training']['generator_path'], map_location=device))

    # ================== Generation with improved sampling ==================
    sos_id = cfg['generator']['num_embeddings']
    seq = torch.full((1, 1), sos_id, dtype=torch.long, device=device)

    print("Starting autoregressive generation...")
    with torch.inference_mode():
        for i in tqdm(range(total_tokens), desc="Generating tokens", unit="token"):
            context = seq[:, -max_context:]

            logits = transformer(context)
            next_token_logits = logits[:, -1, :]

            # === Temperature + Top-K Sampling ===
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                # Remove low-probability tokens
                topk_vals, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < topk_vals[..., -1:]] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            seq = torch.cat([seq, next_token], dim=1)

    # ================== Decoding ==================
    print("\nDecoding tokens to waveform (chunked)...")
    final_indices = seq[:, 1:]  # remove SOS
    chunk_size = 1024
    audio_chunks = []

    for start in tqdm(range(0, final_indices.size(1), chunk_size), desc="Decoding chunks"):
        chunk = final_indices[:, start:start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_slice = vqvae.decoder(z_q).cpu()
        audio_chunks.append(audio_slice)

    generated_waveform = torch.cat(audio_chunks, dim=-1).squeeze(0)
    
    torchaudio.save(output_path, generated_waveform, sample_rate)
    print(f"✅ Success! Saved to: {os.path.abspath(output_path)}")


"""
temperature=0.8 ~ 1.0 → Higher = more creative/random
top_k=30 ~ 80 → Lower = more focused on confident tokens

"""
if __name__ == "__main__":
    # You can change these values when calling
    generate_60s(temperature=0.9, top_k=50)