import yaml
import torch
import torchaudio
import os
from tqdm import tqdm

from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Safe Top-k + Nucleus (Top-p) filtering"""
    logits = logits.clone()
    
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits


@torch.no_grad()
def generate_60s(output_path="output_60s_fixed.wav", 
                 temperature=1.05, 
                 top_k=60,
                 top_p=0.92,
                 repetition_penalty=1.15):
    
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    target_duration_sec = cfg['training']['target_generation_sec']

    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")

    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    total_tokens = int(target_duration_sec * sample_rate / stride)
    max_context = 256

    print(f"🎵 Generating {target_duration_sec}s → ~{total_tokens:,} tokens")
    print(f"Sampling: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")

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

    sos_id = cfg['generator']['num_embeddings']  # 512
    seq = torch.full((1, 1), sos_id, dtype=torch.long, device=device)

    print("Starting generation with safe sampling...")
    
    with torch.inference_mode():
        for i in tqdm(range(total_tokens), desc="Generating tokens", unit="tok"):
            context = seq[:, -max_context:]
            
            logits = transformer(context)[:, -1, :]   # [1, vocab_size]

            # Apply repetition penalty (helps against beeps/ticks)
            if repetition_penalty != 1.0 and seq.size(1) > 1:
                for token_id in set(seq[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Temperature scaling
            logits = logits / temperature

            # Safe Top-k + Top-p filtering
            filtered_logits = top_k_top_p_filtering(
                logits, 
                top_k=top_k, 
                top_p=top_p
            )

            # Convert to probabilities
            probs = torch.softmax(filtered_logits, dim=-1)

            # Safety check: if all probabilities are 0 (very rare now), fall back to argmax
            if probs.sum() <= 0 or torch.isnan(probs).any():
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            seq = torch.cat([seq, next_token], dim=1)

    # Decode
    print("\nDecoding tokens to audio (chunked)...")
    final_indices = seq[:, 1:]  # remove SOS
    chunk_size = 1024
    audio_chunks = []

    for start in tqdm(range(0, final_indices.size(1), chunk_size), desc="Decoding"):
        chunk = final_indices[:, start:start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_slice = vqvae.decoder(z_q).cpu()
        audio_chunks.append(audio_slice)

    generated_waveform = torch.cat(audio_chunks, dim=-1).squeeze(0)
    
    torchaudio.save(output_path, generated_waveform, sample_rate)
    print(f"\n✅ Success! Saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # Recommended settings to reduce beeps/ticks
    generate_60s(
        temperature=1.08,
        top_k=70,
        top_p=0.93,
        repetition_penalty=1.18
    )
