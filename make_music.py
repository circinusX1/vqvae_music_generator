#!/usr/bin/env python3
"""
make-music.py — Generate music with optional reference track and genre models.

Examples:
  python make-music.py
  python make-music.py --gen amon --dur 20
  python make-music.py --ref ./DNL/amon/song.wav --gen amon --dur 30 --temp 0.7
"""

import argparse
import yaml
import torch
import torchaudio
import os
from tqdm import tqdm
import torch.nn.functional as F

from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device
import globals


def load_and_preprocess_reference(file_path, target_sr, target_duration):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    target_samples = int(target_duration * target_sr)
    if waveform.shape[1] > target_samples:
        max_start = waveform.shape[1] - target_samples
        start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        waveform = waveform[:, start:start + target_samples]
    elif waveform.shape[1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
    return waveform.unsqueeze(0)


@torch.no_grad()
def make_music(
    genere="amon",
    ref_path=None,
    target_duration_sec=15,
    temperature=0.65,
    top_k=20,
    repetition_penalty=1.25,
    output_path=None,
):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg["training"]["device"])
    print(f"Device: {device}")
    print(f"Genre: {genere} | Duration: {target_duration_sec}s | temp={temperature} top_k={top_k} rp={repetition_penalty}")

    sample_rate = cfg["dataset"]["sample_rate"]
    stride = cfg["vqvae"]["stride"]
    ref_duration = min(cfg["dataset"].get("duration_sec_train", 6.0), 8.0)

    total_tokens_needed = int(target_duration_sec * sample_rate / stride)
    max_context = 384
    sos_id = cfg["generator"]["num_embeddings"]
    codebook_size = cfg["vqvae"]["num_embeddings"]

    # Prefer best VQ-VAE, then final
    vqvae_path = globals.vae_best_path(genere)
    if not os.path.exists(vqvae_path):
        vqvae_path = globals.vae_path(genere)
    gen_path = globals.gen_model_path(genere)

    if not os.path.exists(vqvae_path):
        print(f"Error: VQ-VAE model not found: {vqvae_path}")
        return
    if not os.path.exists(gen_path):
        print(f"Error: Generator model not found: {gen_path}")
        return

    print(f"Loading VQ-VAE: {vqvae_path}")
    print(f"Loading Generator: {gen_path}")
    vqvae = VQVAEModel(cfg).to(device).eval()
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))

    transformer = MusicTransformer(
        cfg["generator"]["num_embeddings"],
        cfg["generator"]["embedding_dim"],
        cfg["generator"]["hidden_dim"],
        cfg["generator"]["num_layers"],
        cfg["generator"]["num_heads"],
    ).to(device).eval()

    state = torch.load(gen_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    transformer.load_state_dict(state, strict=False)

    sos = torch.full((1, 1), sos_id, dtype=torch.long, device=device)

    if ref_path is not None:
        if not os.path.exists(ref_path):
            print(f"Error: reference file not found: {ref_path}")
            return
        print(f"Encoding reference: {ref_path}")
        ref_wave = load_and_preprocess_reference(ref_path, sample_rate, ref_duration).to(device)
        z_ref = vqvae.encoder(ref_wave)
        _, _, ref_indices = vqvae.quantizer(z_ref)
        print(f"Reference → {ref_indices.shape[1]} tokens")
        seq = torch.cat([sos, ref_indices], dim=1)
    else:
        print("No reference — generating from SOS only")
        seq = sos

    tokens_to_generate = max(1, total_tokens_needed - seq.size(1) + 1)
    print(f"Generating {tokens_to_generate} tokens (~{target_duration_sec}s)...")

    with torch.inference_mode():
        for _ in tqdm(range(tokens_to_generate), desc="Generating"):
            context = seq[:, -max_context:]
            logits = transformer(context)[:, -1, :]

            if repetition_penalty != 1.0 and seq.size(1) > 1:
                for t in set(seq[0, -max_context:].tolist()):
                    logits[0, t] /= repetition_penalty

            logits = logits / temperature
            logits[:, sos_id] = -float("Inf")

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

    print("Decoding...")
    final_indices = seq[:, 1:]
    final_indices = torch.clamp(final_indices, min=0, max=codebook_size - 1)

    chunk_size = 1024
    audio_chunks = []
    for start in tqdm(range(0, final_indices.size(1), chunk_size), desc="Decoding"):
        chunk = final_indices[:, start:start + chunk_size]
        z_q = vqvae.quantizer.embedding(chunk).permute(0, 2, 1).contiguous()
        audio_chunks.append(vqvae.decoder(z_q).cpu())

    waveform = torch.cat(audio_chunks, dim=-1).squeeze(0)

    if output_path is None:
        output_path = f"output_{genere}_{int(target_duration_sec)}s.wav"
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"\n✅ Saved: {os.path.abspath(output_path)}")


def parse_args():
    genres = list(globals.GENERES.keys())
    parser = argparse.ArgumentParser(description="Generate music")
    parser.add_argument("--ref", type=str, default=None, help="Reference song path")
    parser.add_argument("--gen", type=str, default="amon", choices=genres, help="Genre")
    parser.add_argument("--dur", type=float, default=8, help="Duration seconds")
    parser.add_argument("--temp", type=float, default=0.65, help="Temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k")
    parser.add_argument("--rp", type=float, default=1.25, help="Repetition penalty")
    parser.add_argument("--out", type=str, default=None, help="Output wav path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_music(
        genere=args.gen,
        ref_path=args.ref,
        target_duration_sec=args.dur,
        temperature=args.temp,
        top_k=args.top_k,
        repetition_penalty=args.rp,
        output_path=args.out,
    )

