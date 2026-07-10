import yaml
import torch
import torchaudio
import os
from train_vqvae import VQVAEModel
from audio_processing import setup_device

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = setup_device(cfg['training']['device'])
print(f"Using device: {device}")

vqvae = VQVAEModel(cfg).to(device)
vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
vqvae.eval()

# Load first file
dnl_dir = cfg['dataset']['raw_dir']
files = [f for f in os.listdir(dnl_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
if not files:
    print("No audio files found in", dnl_dir)
    exit(1)

ref_path = os.path.join(dnl_dir, files[0])
print(f"Testing reconstruction on: {ref_path}")

wave, sr = torchaudio.load(ref_path)

# Keep on CPU for resampling, then move
if sr != cfg['dataset']['sample_rate']:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg['dataset']['sample_rate'])
    wave = resampler(wave)

wave = wave.mean(0, keepdim=True).unsqueeze(0).to(device)  # Now safe

target_samples = cfg['dataset']['duration_sec'] * cfg['dataset']['sample_rate']
if wave.shape[-1] > target_samples:
    wave = wave[:, :, :target_samples]

print(f"Input shape: {wave.shape}")

with torch.no_grad():
    recon, vq_loss, indices = vqvae(wave)
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Codebook indices shape: {indices.shape} | Unique codes: {torch.unique(indices).numel()}")

torchaudio.save("vqvae_recon_test.wav", recon.cpu().squeeze(0), cfg['dataset']['sample_rate'])
print("✅ Saved: vqvae_recon_test.wav  — Listen to this file!")
