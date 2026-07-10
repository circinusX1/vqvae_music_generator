import yaml
import torch
import torchaudio
import os
from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device
import torch.nn.functional as F

@torch.no_grad()
def generate_fixed(output_path="output_better.wav", target_duration_sec=10, temp=0.85, top_k=25):
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = setup_device(cfg['training']['device'])
    
    # === Reference Song Selection ===
    dnl_path = cfg['dataset']['raw_dir']
    files = [f for f in os.listdir(dnl_path) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    print("\n--- Reference Selection ---")
    for idx, f in enumerate(files):
        print(f"[{idx}] {f}")
    choice = input("Enter index to use as reference (or press Enter to skip): ")
    
    # === Load Models ===
    vqvae = VQVAEModel(cfg).to(device).eval()
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], 
        cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], 
        cfg['generator']['num_layers'], 
        cfg['generator']['num_heads']
    ).to(device).eval()
    transformer.load_state_dict(torch.load(cfg['training']['generator_path'], map_location=device))
    
    # === Encode Reference (if chosen) ===
    ref_latent = None
    if choice.isdigit() and int(choice) < len(files):
        ref_path = os.path.join(dnl_path, files[int(choice)])
        print(f"Using reference: {ref_path}")
        wave, sr = torchaudio.load(ref_path)
        if sr != cfg['dataset']['sample_rate']:
            wave = torchaudio.transforms.Resample(sr, cfg['dataset']['sample_rate'])(wave)
        wave = wave.mean(0, keepdim=True).unsqueeze(0).to(device)
        with torch.no_grad():
            ref_latent = vqvae.encoder(wave).mean(dim=2)  # [1, C]
    
    # === Generation ===
    sos_id = cfg['generator']['num_embeddings']
    seq = torch.full((1, 1), sos_id, dtype=torch.long, device=device)
    
    total_tokens = int(target_duration_sec * cfg['dataset']['sample_rate'] / cfg['vqvae']['stride'])
    max_ctx = 512   # Should match what you used during training
    
    print(f"Generating {total_tokens} tokens | temp={temp} | top_k={top_k}...")
    
    for i in range(total_tokens):
        context = seq[:, -max_ctx:]
        logits = transformer(context, ref_latent=ref_latent)
        
        # Improved sampling
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[..., -1:]] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        seq = torch.cat([seq, next_token], dim=1)
        
        if (i + 1) % 2000 == 0:
            print(f"Progress: {(i+1)/total_tokens*100:.1f}%")
    
    # === Chunked Decoding ===
    final_indices = seq[:, 1:]
    chunk_size = 1024
    audio_chunks = []
    print("Decoding in chunks...")
    for start in range(0, final_indices.size(1), chunk_size):
        chunk = final_indices[:, start:start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_chunks.append(vqvae.decoder(z_q).cpu())
    
    waveform = torch.cat(audio_chunks, dim=-1).squeeze(0)
    torchaudio.save(output_path, waveform, cfg['dataset']['sample_rate'])
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    generate_fixed(temp=1.0, top_k=40)   # ← Change these values
