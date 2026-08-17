import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torchaudio
import yaml
import torch
import gc
import torch.nn as nn
from data_loader import get_dataloader
from encoder import AudioEncoder
from quantizer import VectorQuantizer
from decoder import AudioDecoder
from metrics import MultiScaleSpectralLoss
from audio_processing import setup_device
import globals


def print_vram_usage(milestone_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # print(f"--- VRAM [{milestone_name}] --- Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB | Peak: {max_allocated:.2f}MB\n")


def preprocess_dataset(genere, sample_rate=22050):
    """Standardizes all audio to 22050Hz, Mono, 16-bit PCM WAV."""
    print(f"Starting pre-processing: {genere} -> {globals.gen_proc(genere)}")

    extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    if not os.path.exists(globals.gen_path(genere)):
        os.makedirs(globals.gen_path(genere))
    if not os.path.exists(globals.gen_proc(genere)):
        os.makedirs(globals.gen_proc(genere))

    raw_dir = globals.gen_path(genere)
    processed_dir = globals.gen_proc(genere)

    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(extensions):
                rel_path = ""
                target_folder = os.path.join(processed_dir, rel_path)
                os.makedirs(target_folder, exist_ok=True)

                raw_path = os.path.join(root, file)
                save_path = os.path.join(target_folder, os.path.splitext(file)[0] + ".wav")

                if os.path.exists(save_path):
                    continue

                try:
                    print(f"preparing {raw_path}")
                    waveform, sr = torchaudio.load(raw_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != sample_rate:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                        waveform = resampler(waveform)
                    torchaudio.save(save_path, waveform, sample_rate, encoding='PCM_S', bits_per_sample=16)
                except Exception as e:
                    print(f"Failed to process {raw_path}: {e}")
    print("Pre-processing complete.")


class VQVAEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AudioEncoder(
            cfg['vqvae']['in_channels'],
            cfg['vqvae']['hidden_channels'],
            cfg['vqvae']['residual_channels'],
            cfg['vqvae']['num_residual_layers'],
            cfg['vqvae']['stride'],
            embedding_dim=cfg['vqvae']['embedding_dim']
        )
        self.quantizer = VectorQuantizer(
            cfg['vqvae']['num_embeddings'], cfg['vqvae']['embedding_dim'], cfg['vqvae']['commitment_cost']
        )
        self.decoder = AudioDecoder(
            cfg['vqvae']['in_channels'],
            cfg['vqvae']['embedding_dim'],
            cfg['vqvae']['hidden_channels'],
            cfg['vqvae']['residual_channels'],
            cfg['vqvae']['num_residual_layers'],
            cfg['vqvae']['stride']
        )

    def forward(self, x):
        target_length = x.shape[-1]
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        if x_recon.shape[-1] != target_length:
            x_recon = torch.nn.functional.interpolate(
                x_recon, size=target_length, mode='linear', align_corners=False
            )
        return x_recon, vq_loss, indices


def sub_main(genere):
    wav_seconds = 0
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    preprocess_dataset(genere, cfg['dataset']['sample_rate'])

    chunks_per_file = cfg['dataset'].get('train_chunks_per_file', 1)
    loader = get_dataloader(
        globals.gen_proc(genere),
        cfg['training']['vqvae_batch_size'],
        cfg['dataset']['sample_rate'],
        cfg['dataset']['duration_sec_train'],
        chunks_per_file=chunks_per_file,
    )
    print(f"Dataset size: {len(loader.dataset)} samples "
          f"({len(loader.dataset) // max(chunks_per_file, 1)} files × {chunks_per_file} chunks)")

    device = setup_device(cfg['training']['device'])

    if len(loader) == 0:
        print("Error: VQ-VAE DataLoader yielded 0 tracks. Verify nested files exist inside subfolders.")
        return

    start_epoch = 0
    checkpoint_path = globals.vae_chk_path(genere)
    best_path = globals.vae_best_path(genere)
    final_path = globals.vae_path(genere)

    model = VQVAEModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    # Optional step LR decay (lr_step_size <= 0 disables)
    lr_step_size = int(cfg['training'].get('lr_step_size', 20))
    lr_gamma = float(cfg['training'].get('lr_gamma', 0.5))
    scheduler = None
    if lr_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        print(f"LR schedule: StepLR every {lr_step_size} epochs, gamma={lr_gamma}")

    # Early stopping: stop if no improvement for N epochs
    early_stop_patience = int(cfg['training'].get('early_stop_patience', 10))
    epochs_without_improve = 0

    best_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        epochs_without_improve = checkpoint.get('epochs_without_improve', 0)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    criterion = MultiScaleSpectralLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')
    grad_clip = float(cfg['training'].get('grad_clip', 1.0))

    plotter = globals.Ploter()
    plotter.ax.set_title(f'Live VQ-VAE Loss — {genere}')

    print("Beginning VQ-VAE Stage 1 Training...")
    for epoch in range(start_epoch, cfg['training']['vqvae_epochs']):
        torch.cuda.empty_cache()
        gc.collect()
        total_loss = 0.0
        total_batches = len(loader)

        for batch_idx, batch in enumerate(loader):
            wav_seconds += batch.shape[-1] / cfg['dataset']['sample_rate']
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                x_recon, vq_loss, _ = model(batch)
                recon_loss = criterion(batch, x_recon)
                loss = recon_loss + vq_loss

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at batch {batch_idx}: recon={float(recon_loss):.4f} vq={float(vq_loss):.4f}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = ((batch_idx + 1) / total_batches) * 100
                # print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        plotter.add_point(avg_loss)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg['training']['lr']

        improved = avg_loss < best_loss
        if improved:
            best_loss = avg_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"Epoch {epoch+1}/{cfg['training']['vqvae_epochs']} - Loss: {avg_loss:.4f}  LR: {current_lr:.2e}  ★ best → {best_path}")
        else:
            epochs_without_improve += 1
            print(f"Epoch {epoch+1}/{cfg['training']['vqvae_epochs']} - Loss: {avg_loss:.4f}  LR: {current_lr:.2e}  (best: {best_loss:.4f}, no improve: {epochs_without_improve}/{early_stop_patience})")

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'epochs_without_improve': epochs_without_improve,
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(ckpt, checkpoint_path)

        print_vram_usage(f"Epoch {epoch+1} Complete")

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(f"Early stopping: no improvement for {early_stop_patience} epochs (best loss={best_loss:.4f})")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best weights (loss={best_loss:.4f}) for final export")
    torch.save(model.state_dict(), final_path)
    print(f"=== Success! Best VQ-VAE saved to: {os.path.abspath(final_path)} ===")
    print(f"    Best checkpoint also at: {os.path.abspath(best_path)}")
    del plotter
    return wav_seconds

def main():
    for genere, query in globals.GENERES.items():
        search_term, number_of_songs_to_download, enabled = query.split("|")
        if enabled == '0':
            continue
        print(f"Starting VQ-VAE training for genere: {genere} (Query: {query})")
        
        seconds = sub_main(genere)
        print(f"Training complete for genere: {genere} (Query: {query}) for wav: {seconds} seconds")


if __name__ == "__main__":
    main()



