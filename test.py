import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import torchaudio

import globals
from train_vqvae import VQVAEModel
from audio_processing import setup_device


def preprocess_reference_audio(file_path, target_sample_rate=22050):
    """Loads and standardizes an audio file to mono PCM at target sample rate."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reference audio file not found: {file_path}")

    waveform, sr = torchaudio.load(file_path)

    # Convert multi-channel to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to target sample rate if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform


def calculate_match_percentage(original, reconstructed):
    """
    Calculates percentage match between the reference song and reconstructed song
    using spectral magnitude cosine similarity.
    """
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    orig = original[..., :min_len].detach().cpu().float()
    recon = reconstructed[..., :min_len].detach().cpu().float()

    # Pass Hann window to eliminate spectral leakage warning
    window = torch.hann_window(2048)

    # Compute Short-Time Fourier Transform (STFT) magnitudes
    stft_orig = torch.abs(
        torch.stft(orig.squeeze(0), n_fft=2048, hop_length=512, window=window, return_complex=True)
    )
    stft_recon = torch.abs(
        torch.stft(recon.squeeze(0), n_fft=2048, hop_length=512, window=window, return_complex=True)
    )

    # Use .reshape(1, -1) instead of .view(1, -1) for non-contiguous tensor layout
    cosine_sim = F.cosine_similarity(stft_orig.reshape(1, -1), stft_recon.reshape(1, -1)).item()
    
    # Scale to percentage [0% - 100%]
    match_percentage = max(0.0, cosine_sim) * 100.0
    return match_percentage


def test_vqvae_reconstruction(
    genere="all",
    reference_song="DNL/goodbye/cage.wav",
    model_path="M/all-vqvae.pt",
    output_path="xsong.wav",
):
    """
    Tests VQ-VAE reconstruction on a reference song and prints the match percentage.
    """
    
    # output_path= f"x_{reference_song}"
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        return

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg["training"]["device"])
    sample_rate = cfg["dataset"]["sample_rate"]

    # Locate model checkpoint
    if model_path is None:
        model_path = globals.vae_best_path(genere)
        if not os.path.exists(model_path):
            model_path = globals.vae_path(genere)

    if not os.path.exists(model_path):
        print(f"Error: VQ-VAE model checkpoint not found at '{model_path}'")
        return

    # Load model and weights
    print(f"Loading VQ-VAE model from: {model_path}")
    model = VQVAEModel(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Fallback to first processed song in genre directory if reference song does not exist
    if reference_song is None or not os.path.exists(reference_song):
        proc_dir = globals.gen_proc(genere)
        if os.path.exists(proc_dir):
            files = [os.path.join(proc_dir, f) for f in os.listdir(proc_dir) if f.endswith(".wav")]
            if files:
                reference_song = files[0]

    if reference_song is None or not os.path.exists(reference_song):
        print(f"Error: Reference song '{reference_song}' not found.")
        return

    # Load audio
    print(f"Loading reference song: {reference_song}")
    ref_waveform = preprocess_reference_audio(reference_song, target_sample_rate=sample_rate)
    input_tensor = ref_waveform.unsqueeze(0).to(device)  # Shape: (1, 1, samples)

    # Perform VQ-VAE reconstruction pass
    with torch.no_grad():
        recon_tensor, _, _ = model(input_tensor)

    # Calculate and output match percentage
    match_pct = calculate_match_percentage(input_tensor, recon_tensor)

    print("\n" + "=" * 50)
    print(f" Reference Song: {os.path.basename(reference_song)}")
    print(f" Genre:          {genere}")
    print(f" Match Score:    {match_pct:.2f}%")
    print("=" * 50 + "\n")

    # Save output reconstructed audio if specified
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(output_path, recon_tensor.squeeze(0).cpu(), sample_rate)
        print(f"Saved reconstructed audio to: {output_path}")

    return match_pct


def main():
    parser = argparse.ArgumentParser(description="Test VQ-VAE Song Reconstruction Match")

    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default="DNL/goodbye/cage.wav",
        help="Path to reference song file (default: 'DNL/goodbye/cage.wav')",
    )
    parser.add_argument(
        "--genre",
        "-g",
        type=str,
        default="all",
        help="Genre key to load model weights (default: 'all')",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="M/all-vqvae.pt",
        help="Path to model checkpoint (default: 'M/all-vqvae.pt')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="xsong.wav",
        help="Path to save reconstructed output WAV (default: 'xsong.wav')",
    )

    args = parser.parse_args()

    test_vqvae_reconstruction(
        genere=args.genre,
        reference_song=args.reference,
        model_path=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
    
