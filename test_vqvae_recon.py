import yaml
import torch
import torchaudio
import os
from train_vqvae import VQVAEModel
from audio_processing import setup_device
import globals


def test_vqvae_reconstruction(genere):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")

    vqvae = VQVAEModel(cfg).to(device)
    # vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae_path = globals.vae_best_path(genere)



    if not os.path.exists(vqvae_path): 
        print(f"Checkpoint not found at {vqvae_path}")
    else:
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        vqvae.eval()
    # ← PUT THE PRINT HERE
    print("Codebook usage:", vqvae.quantizer.get_codebook_usage(), "/", vqvae.quantizer.num_embeddings)


    # Load first file
    dnl_dir = globals.gen_proc("all")
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

    torchaudio.save(f"{genere}-test.wav", recon.cpu().squeeze(0), cfg['dataset']['sample_rate'])
    print(f"✅ Saved: {genere}-test.wav  — Listen to this file!")


if __name__ == "__main__":
    for genere, query in globals.GENERES.items():
        search_term, number_of_songs_to_download, enabled = query.split("|")
        if enabled == '0':
            print(f"Skipping disabled genre: {genere}")
            continue
        test_vqvae_reconstruction(genere)