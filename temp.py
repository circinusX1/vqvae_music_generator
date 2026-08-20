# train_generator.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import config
from utils import Ploter
from dataset import SyntheticVibrationDataset
from models import (
    MultiModalConvVAE,
    preprocess_batch_gpu
)

# ==========================================
# CUDA & PERFORMANCE OPTIMIZATIONS (~70% VRAM)
# ==========================================
torch.set_float32_matmul_precision('high')  # Enables TF32 on modern GPUs
torch.backends.cudnn.benchmark = True       # Autotunes convolution algorithms


class MultiResolutionSTFTLoss(nn.Module):
    """
    Spectral convergence + log STFT magnitude loss across multiple FFT frame sizes.
    Prevents loss flatlining by constraining both time-domain and frequency-domain details.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        loss = 0.0
        # Flatten spatial/channel dimensions if 4D
        if x.dim() == 4:
            x = x.squeeze(1)
            y = y.squeeze(1)

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length).to(x.device)
            x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True).abs()
            y_stft = torch.stft(y, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True).abs()

            # Spectral Convergence Loss
            sc_loss = torch.norm(y_stft - x_stft, p="fro") / (torch.norm(y_stft, p="fro") + 1e-7)
            # Log STFT Magnitude Loss
            mag_loss = F.l1_loss(torch.log(x_stft + 1e-5), torch.log(y_stft + 1e-5))

            loss += sc_loss + mag_loss

        return loss / len(self.fft_sizes)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    os.makedirs("M", exist_ok=True)

    # 1. Dataset & High-Throughput Loaders
    full_dataset = SyntheticVibrationDataset(num_samples=config.NUM_SAMPLES, seed=42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 2. Generator Model Setup
    generator = MultiModalConvVAE(latent_dim=config.LATENT_SIZE).to(device)

    # Resume from existing checkpoint if present
    ckpt_path = "M/gen_best.pt" if os.path.exists("M/gen_best.pt") else ("M/gen_final.pt" if os.path.exists("M/gen_final.pt") else None)
    if ckpt_path:
        generator.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded existing generator checkpoint from '{ckpt_path}'. Continuing training...")

    # Optimized AdamW with lower beta1 for generator stability
    optimizer = torch.optim.AdamW(generator.parameters(), lr=getattr(config, 'GEN_LR', 2e-3), betas=(0.5, 0.9), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.VAE_EPOCHS, eta_min=1e-5)

    stft_criterion = MultiResolutionSTFTLoss().to(device)
    plotter = Ploter(title="Generator Training Loss (Spectral + L1)")

    best_loss = float('inf')
    patience_counter = 0

    print("\n--- Training Generator Network ---")
    generator.train()

    for epoch in range(config.VAE_EPOCHS):
        running_total, running_l1, running_stft = 0.0, 0.0, 0.0

        for batch_vib, batch_temp, _ in train_loader:
            x_vib, x_temp = preprocess_batch_gpu(batch_vib, batch_temp, device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            recon_vib, recon_temp, mu, logvar = generator(x_vib, x_temp)

            # Composite Loss Formulation (L1 + Multi-Resolution Spectral Loss)
            l1_loss = F.l1_loss(recon_vib, x_vib) + 0.1 * F.l1_loss(recon_temp, x_temp)
            stft_loss = stft_criterion(recon_vib, x_vib)
            
            # KL Divergence with Annealing
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            beta = min(1.0, (epoch + 1) / 10.0)  # Smooth KL warmup over first 10 epochs

            total_loss = l1_loss + (0.5 * stft_loss) + (beta * 0.01 * kl_loss)

            total_loss.backward()
            
            # Gradient clipping to prevent flatlining and exploding gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            
            optimizer.step()

            running_total += total_loss.item()
            running_l1 += l1_loss.item()
            running_stft += stft_loss.item()

        scheduler.step()

        num_batches = len(train_loader)
        avg_total = running_total / num_batches
        avg_l1 = running_l1 / num_batches
        avg_stft = running_stft / num_batches

        plotter.add_point(epoch + 1, avg_total, avg_l1, avg_stft)
        print(f"Epoch [{epoch+1:2d}/{config.VAE_EPOCHS}] | Total: {avg_total:.4f} | L1: {avg_l1:.4f} | STFT: {avg_stft:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        curr_epoch = epoch + 1

        # Checkpoint: Best Model
        if avg_total < best_loss:
            best_loss = avg_total
            patience_counter = 0
            torch.save(generator.state_dict(), "M/gen_best.pt")
            print(f"  --> Saved new best Generator to M/gen_best.pt (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1

        # Checkpoint: Milestone
        if config.MILES > 0 and curr_epoch % config.MILES == 0:
            torch.save(generator.state_dict(), "M/gen_mile.pt")
            print(f"  --> Saved milestone checkpoint to M/gen_mile.pt")

        # Early Stopping Check
        if config.VAE_EARLY > 0 and patience_counter >= config.VAE_EARLY:
            print(f"--> Early Stopping triggered at epoch {curr_epoch}.")
            break

    # Save Final Model
    torch.save(generator.state_dict(), "M/gen_final.pt")
    print("Saved final Generator checkpoint to M/gen_final.pt")


if __name__ == "__main__":
    main()
    