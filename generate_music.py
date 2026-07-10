import yaml
import torch
import torchaudio
import os
from tqdm import tqdm

from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device


@torch.no_grad()
def generate_music(output_path="output_15s_better.wav", 
                   target_duration_sec=15, 
                   temperature=0.85, 
                   top_k=40,
                   repetition_penalty=1.08):
    
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")

    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    total_tokens = int(target_duration_sec * sample_rate / stride)
    max_context = 256

    print(f"Generating {target_duration_sec}s → ~{total_tokens:,} tokens")

    # Load models
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

    sos_id = cfg['generator']['num_embeddings']
    seq = torch.full((1, 1), sos_id, dtype=torch.long, device=device)

    print("Generating with conservative settings...")
    with torch.inference_mode():
        for i in tqdm(range(total_tokens), desc="Generating"):
            context = seq[:, -max_context:]
            logits = transformer(context)[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0 and seq.size(1) > 1:
                for token_id in set(seq[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            logits = logits / temperature

            # Conservative Top-k only (no top-p for stability)
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            seq = torch.cat([seq, next_token], dim=1)

    # Decode
    print("Decoding...")
    final_indices = seq[:, 1:]
    chunk_size = 1024
    audio_chunks = []

    for start in tqdm(range(0, final_indices.size(1), chunk_size), desc="Decoding"):
        chunk = final_indices[:, start:start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_chunks.append(vqvae.decoder(z_q).cpu())

    waveform = torch.cat(audio_chunks, dim=-1).squeeze(0)
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    generate_music(
        target_duration_sec=5,     # Start with 15s
        temperature=0.85,
        top_k=40,
        repetition_penalty=1.08
    )
    