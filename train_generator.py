import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
from data_loader import get_dataloader
from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device

def print_vram_usage(milestone_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"> {milestone_name}: allocated: {allocated:.2f}MB, reserved: {reserved:.2f}MB, peak: {max_allocated:.2f}MB\n")

def main():
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = setup_device(cfg['training']['device'])
    print(f"Using device: {device}")
    
    # Load Frozen VQ-VAE
    vqvae = VQVAEModel(cfg).to(device)
    if not os.path.exists(cfg['training']['vqvae_path']):
        print("Error: VQ-VAE not found. Train it first.")
        return
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    
    # Shorter duration for stable generator training
    train_duration = 1.5
    loader = get_dataloader(
        cfg['dataset']['processed_dir'], 
        cfg['training']['gen_batch_size'],
        cfg['dataset']['sample_rate'], 
        train_duration
    )
    
    if len(loader) == 0:
        print("No data found!")
        return

    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], 
        cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], 
        cfg['generator']['num_layers'], 
        cfg['generator']['num_heads']
    ).to(device)
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg['training']['lr'])
    criterion = nn.CrossEntropyLoss()
    sos_token_id = cfg['generator']['num_embeddings']
    
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = cfg['generator'].get('accumulation_steps', 8)
    
    checkpoint_path = cfg['training']['generator_path']
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            transformer.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            # Old checkpoint format
            transformer.load_state_dict(checkpoint)
    
    print(f"Starting from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, cfg['training']['gen_epochs']):
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            
            with torch.no_grad():
                z = vqvae.encoder(batch)
                _, _, indices = vqvae.quantizer(z)
            
            # Short context window
            max_ctx = 1024
            if indices.size(1) > max_ctx:
                start = torch.randint(0, indices.size(1) - max_ctx + 1, (1,)).item()
                indices = indices[:, start:start + max_ctx]
            
            sos_tokens = torch.full((indices.size(0), 1), sos_token_id, dtype=torch.long, device=device)
            inputs = torch.cat([sos_tokens, indices[:, :-1]], dim=1)
            targets = indices
            
            # Mixed precision + Label Smoothing
            with torch.amp.autocast('cuda'):
                logits = transformer(inputs)
                # Label smoothing
                loss = F.kl_div(
                    F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1),
                    F.one_hot(targets.view(-1), num_classes=logits.size(-1)).float().to(device) * 0.9 + 0.1 / logits.size(-1),
                    reduction='batchmean'
                )
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)   # ← Fixed
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            # Cleanup
            del batch, z, indices, sos_tokens, inputs, targets, logits
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg['training']['gen_epochs']} - Avg Loss: {avg_loss:.4f}")
        print_vram_usage(f"End of Epoch {epoch+1}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    print("=== Generator training finished ===")
    torch.save(transformer.state_dict(), cfg['training']['generator_path'])

if __name__ == "__main__":
    main()

    