import yaml
import torch
import torchaudio
import os
from train_vqvae import VQVAEModel
from audio_processing import setup_device
import globals


def compute_reconstruction_percentage(target: torch.Tensor, recon: torch.Tensor) -> dict:
    """
    Compute how closely the reconstructed waveform matches the reference.
    Returns a dict with MAE and match percentage (based on cosine similarity).
    """
    # Flatten to 1-D for metrics
    t = target.detach().float().flatten()
    r = recon.detach().float().flatten()

    # Align lengths (safety)
    min_len = min(t.numel(), r.numel())
    t = t[:min_len]
    r = r[:min_len]

    # Mean Absolute Error
    mae = torch.mean(torch.abs(t - r)).item()

    # Center both signals (removes DC offset bias)
    t_c = t - t.mean()
    r_c = r - r.mean()

    # Cosine similarity → percentage match
    denom = (t_c.norm() * r_c.norm()) + 1e-8
    cosine_sim = torch.dot(t_c, r_c) / denom
    match_pct = max(0.0, min(100.0, cosine_sim.item() * 100.0))

    # Relative energy error (optional secondary view)
    energy_t = torch.mean(t ** 2) + 1e-8
    energy_err = torch.mean((t - r) ** 2) / energy_t
    energy_match_pct = max(0.0, min(100.0, (1.0 - energy_err.item()) * 100.0))

    return {
        "mae": mae,
        "match_pct": match_pct,
        "energy_match_pct": energy_match_pct,
    }


def test_vqvae_reconstruction(genere):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")

    vqvae = VQVAEModel(cfg).to(device)
    vqvae_path = globals.vae_best_path(genere)

    if not os.path.exists(vqvae_path):
        print(f"Checkpoint not found at {vqvae_path}")
        return
    else:
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        vqvae.eval()

    # Load first available processed file for this genre
    dnl_dir = globals.gen_proc(genere)
    if not os.path.isdir(dnl_dir):
        print(f"Processed directory not found: {dnl_dir}")
        return

    files = [f for f in os.listdir(dnl_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    if not files:
        print("No audio files found in", dnl_dir)
        return

    ref_path = os.path.join(dnl_dir, files[2])
    print(f"Testing reconstruction on: {ref_path}")

    wave, sr = torchaudio.load(ref_path)

    # Resample on CPU first
    if sr != cfg['dataset']['sample_rate']:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg['dataset']['sample_rate'])
        wave = resampler(wave)

    wave = wave.mean(0, keepdim=True).unsqueeze(0).to(device)

    target_samples = int(cfg['dataset']['duration_sec'] * cfg['dataset']['sample_rate'])
    if wave.shape[-1] > target_samples:
        wave = wave[:, :, :target_samples]
    elif wave.shape[-1] < target_samples:
        pad = target_samples - wave.shape[-1]
        wave = torch.nn.functional.pad(wave, (0, pad))

    print(f"Input shape: {wave.shape}")

    with torch.no_grad():
        recon, vq_loss, indices = vqvae(wave)
        print(f"VQ Loss: {vq_loss.item():.4f}")
        print(f"Codebook indices shape: {indices.shape} | Unique codes: {torch.unique(indices).numel()}")

        # ---- Reconstruction match percentage ----
        metrics = compute_reconstruction_percentage(wave, recon)
        print(f"Reconstruction MAE          : {metrics['mae']:.6f}")
        print(f"Match percentage (cosine)   : {metrics['match_pct']:.1f}%")
        print(f"Energy match percentage     : {metrics['energy_match_pct']:.1f}%")
        print(f"→ {genere}-test.wav matches the reference at {metrics['match_pct']:.1f}%")

    out_name = f"{genere}-test.wav"
    torchaudio.save(out_name, recon.cpu().squeeze(0), cfg['dataset']['sample_rate'])
    print(f"✅ Saved: {out_name}  — Listen to this file!")


if __name__ == "__main__":
    for genere, query in globals.GENERES.items():
        parts = query.split("|")
        if len(parts) >= 3:
            enabled = parts[2]
        else:
            enabled = "1"
        if enabled == "0":
            print(f"Skipping disabled genre: {genere}")
            continue
        test_vqvae_reconstruction(genere)

