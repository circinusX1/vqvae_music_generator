import yaml
import torch
import gc
import torch.nn as nn
import os

from src.data_loader import get_dataloader
from src.models.encoder import AudioEncoder
from src.models.quantizer import VectorQuantizer
from src.models.decoder import AudioDecoder
from src.utils.metrics import MultiScaleSpectralLoss
from src.utils.audio_processing import setup_device

def print_vram_usage(milestone_name):
    """Helper to print current and max VRAM utilization in Megabytes"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"--- VRAM [{milestone_name}] --- Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB | Peak: {max_allocated:.2f}MB\n")

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
        
    device = setup_device(cfg['training']['device'])
    
    # Structural check to help ensure your dataset engine crawls nested subdirectories
    print(f"Scanning subdirectories under target workspace: {cfg['dataset']['raw_dir']}")
    for genre in cfg.get('genres', {}).keys():
        genre_path = os.path.join(cfg['dataset']['raw_dir'], genre)
        if os.path.exists(genre_path):
            file_count = len([f for f in os.listdir(genre_path) if f.endswith(('.wav', '.mp3'))])
            print(f"  -> Found Subfolder: '{genre}' containing {file_count} tracks.")
        else:
            print(f"{genre_path}  no such folder")    
            
    loader = get_dataloader(
        cfg['dataset']['raw_dir'], cfg['training']['vqvae_batch_size'],
        cfg['dataset']['sample_rate'], cfg['dataset']['duration_sec']
    )
    
    if len(loader) == 0:
        print("Error: VQ-VAE DataLoader yielded 0 tracks. Verify nested files exist inside subfolders.")
        return
    
    model = VQVAEModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = MultiScaleSpectralLoss()
    
    # AMP Scaling configurations to maximize VRAM throughput on your 8GB GPU
    scaler = torch.amp.GradScaler('cuda')
    
    print("Beginning VQ-VAE Stage 1 Training...")
    for epoch in range(cfg['training']['vqvae_epochs']):
        torch.cuda.empty_cache()
        gc.collect()
        total_loss = 0
        
        for batch in loader:
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
            
        print(f"Epoch {epoch+1}/{cfg['training']['vqvae_epochs']} - Loss: {total_loss/len(loader):.4f}")
        print_vram_usage(f"Epoch {epoch+1} Complete")

    torch.save(model.state_dict(), cfg['training']['vqvae_path'])
    print(f"=== Success! VQ-VAE weights successfully saved to: {os.path.abspath(cfg['training']['vqvae_path'])} ===")

if __name__ == "__main__":
    main()
