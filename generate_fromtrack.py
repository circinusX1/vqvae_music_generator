import yaml
import torch
import torchaudio
import os
from tqdm import tqdm
import torch.nn.functional as F

from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device


def load_and_preprocess_reference(file_path, target_sr, target_duration):
    """Load reference audio and prepare it exactly like training data"""
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Crop or pad to target duration
    target_samples = int(target_duration * target_sr)
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        pad_amount = target_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_amount))

    return waveform.unsqueeze(0)  # [1, 1, T]


@torch.no_grad()
def generate_from_track(output_path="output_fromtrack.wav",
                        target_duration_sec=30,
                        temperature=0.9,
                        top_k=50,
                        repetition_penalty=1.1):
    
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")

    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    ref_duration = cfg['dataset'].get('duration_sec_train', 1.5)  # Use training duration

    total_tokens_needed = int(target_duration_sec * sample_rate / stride)
    max_context = 256

    # ==================== Reference Track Selection ====================
    dnl_path = cfg['dataset']['raw_dir']
    files = [f for f in os.listdir(dnl_path) 
             if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    if not files:
        print(f"No audio files found in {dnl_path}")
        return

    print("\n=== Available Reference Tracks ===")
    for idx, f in enumerate(files):
        print(f"[{idx}] {f}")

    choice = input("\nEnter the index of the reference track to use: ").strip()

    try:
        ref_idx = int(choice)
        if ref_idx < 0 or ref_idx >= len(files):
            raise ValueError
    except:
        print("Invalid choice. Exiting.")
        return

    ref_path = os.path.join(dnl_path, files[ref_idx])
    print(f"\nUsing reference: {ref_path}")

    # ==================== Load Models ====================
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
    transformer.load_state_dict(torch.load(cfg["training"]["generator_path"], map_location=device), strict=False)

    # ==================== Encode Reference Track ====================
    print("Encoding reference track into tokens...")
    ref_wave = load_and_preprocess_reference(ref_path, sample_rate, ref_duration).to(device)

    with torch.no_grad():
        z_ref = vqvae.encoder(ref_wave)
        _, _, ref_indices = vqvae.quantizer(z_ref)  # [1, T_ref_tokens]

    print(f"Reference encoded into {ref_indices.shape[1]} tokens")

    # ==================== Start Generation from Reference ====================
    sos_id = cfg['generator']['num_embeddings']
    sos_tokens = torch.full((1, 1), sos_id, dtype=torch.long, device=device)
    
    # Prime the sequence with reference tokens
    generated_sequence = torch.cat([sos_tokens, ref_indices], dim=1)

    tokens_already_generated = generated_sequence.size(1) - 1
    tokens_to_generate = total_tokens_needed - tokens_already_generated

    print(f"Primed with {tokens_already_generated} tokens from reference.")
    print(f"Will generate {tokens_to_generate} additional tokens...")

    with torch.inference_mode():
        for i in tqdm(range(tokens_to_generate), desc="Generating continuation"):
            context = generated_sequence[:, -max_context:]

            logits = transformer(context)[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_sequence[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            logits = logits / temperature

            # Ban SOS token (512) so it is never generated again
            logits[:, sos_id] = -float("Inf")

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

    # ==================== Decode ====================
    print("\nDecoding final tokens to audio...")
    final_indices = generated_sequence[:, 1:]  # remove SOS

    # HARD SAFETY CLAMP - prevents index out of bounds crash
    codebook_size = cfg["vqvae"]["num_embeddings"]
    final_indices = torch.clamp(final_indices, min=0, max=codebook_size - 1)
    print(f"Clamped indices to range 0..{codebook_size-1}")

    chunk_size = 1024
    audio_chunks = []

    for start in tqdm(range(0, final_indices.size(1), chunk_size), desc="Decoding"):
        chunk = final_indices[:, start:start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_slice = vqvae.decoder(z_q).cpu()
        audio_chunks.append(audio_slice)

    generated_waveform = torch.cat(audio_chunks, dim=-1).squeeze(0)
    
    torchaudio.save(output_path, generated_waveform, sample_rate)
    print(f"\n✅ Success! Generated from reference track → {os.path.abspath(output_path)}")


if __name__ == "__main__":
    generate_from_track(
        target_duration_sec=30,
        temperature=0.9,
        top_k=50,
        repetition_penalty=1.1
    )
