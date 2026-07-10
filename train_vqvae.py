import torchaudio
import yaml
import torch
import gc
import torch.nn as nn
import os
from data_loader import get_dataloader
from encoder import AudioEncoder
from quantizer import VectorQuantizer
from decoder import AudioDecoder
from metrics import MultiScaleSpectralLoss
from audio_processing import setup_device

def print_vram_usage(milestone_name):
    """Helper to print current and max VRAM utilization in Megabytes"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"--- VRAM [{milestone_name}] --- Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB | Peak: {max_allocated:.2f}MB\n")


import shutil

def preprocess_dataset(raw_dir, processed_dir, sample_rate=22050):
    """Standardizes all audio to 22050Hz, Mono, 16-bit PCM WAV."""
    print(f"Starting pre-processing: {raw_dir} -> {processed_dir}")
    
    # Supported formats
    extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(extensions):
                # Create relative path to maintain folder structure (e.g., metal_amon/track.mp3)
                rel_path = os.path.relpath(root, raw_dir)
                target_folder = os.path.join(processed_dir, rel_path)
                os.makedirs(target_folder, exist_ok=True)
                
                raw_path = os.path.join(root, file)
                save_path = os.path.join(target_folder, os.path.splitext(file)[0] + ".wav")
                
                if os.path.exists(save_path):
                    continue # Skip already processed files
                
                try:
                    print(f"preparing {raw_path}")
                    waveform, sr = torchaudio.load(raw_path)
                    
                    # Convert to Mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                        
                    # Resample
                    if sr != sample_rate:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                        waveform = resampler(waveform)
                        
                    # Save as 16-bit PCM WAV
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
        # 1. Keep track of the original input length (e.g., 88200)
        target_length = x.shape[-1] 
        
        # 2. Run the VQ-VAE pipeline
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        
        # 3. Dynamic slice/pad to guarantee x_recon matches x perfectly
        if x_recon.shape[-1] != target_length:
            x_recon = torch.nn.functional.interpolate(
                x_recon, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )
            
        return x_recon, vq_loss, indices
    
def main():
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    
    # TRIGGER PRE-PROCESSING
    # This runs once and creates the standardized data
    preprocess_dataset(cfg['dataset']['raw_dir'], cfg['dataset']['processed_dir'], cfg['dataset']['sample_rate'])
    
    # CRITICAL: Point the loader to the processed directory now
    loader = get_dataloader(
        cfg['dataset']['processed_dir'], # Changed from raw_dir to processed_dir
        cfg['training']['vqvae_batch_size'],
        cfg['dataset']['sample_rate'], cfg['dataset']['duration_sec_train']
    )        
        
    device = setup_device(cfg['training']['device'])
    
    ll = len(loader)
    if ll == 0:
        print("Error: VQ-VAE DataLoader yielded 0 tracks. Verify nested files exist inside subfolders.")
        return
    
    
    # add checkpoints for 10000 files
    start_epoch = 0
    checkpoint_path = "vqvae_checkpoint.pt"
    model = VQVAEModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    
    criterion = MultiScaleSpectralLoss()
    
    # AMP Scaling configurations to maximize VRAM throughput on your 8GB GPU
    scaler = torch.amp.GradScaler('cuda')
    
    print("Beginning VQ-VAE Stage 1 Training...")
    for epoch in range(cfg['training']['vqvae_epochs']):
        torch.cuda.empty_cache()
        gc.collect()
        total_loss = 0
        total_batches = len(loader)
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Autocast handles down-scaling internal layer weights to float16 seamlessly
            with torch.amp.autocast('cuda'):
                x_recon, vq_loss, _ = model(batch)
                recon_loss = criterion(batch, x_recon)
                loss = recon_loss + vq_loss
            
            # Backpropagation using the gradient scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = ((batch_idx + 1) / total_batches) * 100
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | Loss: {loss.item():.4f}")
        # ----------------------------------------

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, checkpoint_path)


        print(f"Epoch {epoch+1}/{cfg['training']['vqvae_epochs']} - Loss: {total_loss/len(loader):.4f}")
        print_vram_usage(f"Epoch {epoch+1} Complete")

    torch.save(model.state_dict(), cfg['training']['vqvae_path'])
    print(f"=== Success! VQ-VAE weights successfully saved to: {os.path.abspath(cfg['training']['vqvae_path'])} ===")

if __name__ == "__main__":
    main()
