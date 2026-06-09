import yaml
import torch
import torch.nn as nn
import os
import gc
from src.data_loader import get_dataloader
from src.train_vqvae import VQVAEModel
from src.models.generator import MusicTransformer
from src.utils.audio_processing import setup_device
from torch.nn.attention import sdpa_kernel, SDPBackend


def print_vram_usage(milestone_name):
    """Tracks memory growth during training."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"> {milestone_name}: allocated: {allocated:.2f}MB, reserved: {reserved:.2f}MB, peak: {max_allocated:.2f}MB\n")




def main():
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = setup_device(cfg['training']['device'])
    print(f"Using target training device: {device}")
    
    # Load Frozen VQ-VAE model to map raw audio directly to discrete codes
    vqvae = VQVAEModel(cfg).to(device)
    if not os.path.exists(cfg['training']['vqvae_path']):
        print(f"Error: Could not find VQ-VAE weights at {cfg['training']['vqvae_path']}. Train VQ-VAE first.")
        return
        
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae.eval()
    
    loader = get_dataloader(
        cfg['dataset']['processed_dir'], cfg['training']['gen_batch_size'],
        cfg['dataset']['sample_rate'], cfg['dataset']['duration_sec']
    )
    
    if len(loader) == 0:
        print("Error: No training batches generated. Check your data/raw folder.")
        return

    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], cfg['generator']['num_layers'], cfg['generator']['num_heads']
    ).to(device)
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg['training']['lr'])
    criterion = nn.CrossEntropyLoss()
    sos_token_id = cfg['generator']['num_embeddings']  # Index 512
    
    # VRAM Optimization Tools
    scaler = torch.amp.GradScaler('cuda') # Handles gradient scaling for Mixed Precision
    accumulation_steps = 4               # Simulates a larger batch size by updating weights every N steps
    

    checkpoint_path = cfg['training']['generator_path']
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    print(f"From :{start_epoch}  Training ({cfg['training']['gen_epochs']} epochs)...")
    
    try:
        for epoch in range(start_epoch, cfg['training']['gen_epochs']):
            total_loss = 0
            optimizer.zero_grad() # Initialize gradients outside the accumulation step
            
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(device)
                
                with torch.no_grad():
                    z = vqvae.encoder(batch)
                    _, _, indices = vqvae.quantizer(z) # Shape: [B, T_tokens]
                
                # Input sequence: Prefix with SOS token, drop the last time step token
                sos_tokens = torch.full((indices.size(0), 1), sos_token_id, dtype=torch.long, device=device)
                inputs = torch.cat([sos_tokens, indices[:, :-1]], dim=1)
                
                # Target sequence: Expecting the actual original codebook tokens
                targets = indices
                
                # Mixed Precision Forward Pass (Halves VRAM consumption for activations)
                with torch.amp.autocast('cuda'):
                
                    logits = transformer(inputs) 
                    
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / accumulation_steps

                    #   >logits = transformer(inputs) # Expected shape: [B, T, 513]
                    ##  >loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    #    Scale loss to adjust for gradient accumulation
                    ##  >loss = loss / accumulation_steps
                
                # Mixed Precision Backward Pass
                scaler.scale(loss).backward()
                
                # Weight Update step happens every `accumulation_steps` iterations
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                total_loss += loss.item() * accumulation_steps
                
            print(f"Epoch {epoch+1}/{cfg['training']['gen_epochs']} - Average Loss: {total_loss/len(loader):.4f}")
            print(f"Epoch {epoch+1} - Average Loss: {total_loss/len(loader):.4f}")
            print_vram_usage(f"End of Epoch {epoch+1}")
            
            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(loader),
            }, checkpoint_path)
            
            torch.cuda.empty_cache()
            gc.collect()
            
        # Explicitly save to root directory path cleanly
        save_path = cfg['training']['generator_path']
        torch.save(transformer.state_dict(), save_path)
        print(f"=== Success! Prior Transformer weights successfully saved to: {os.path.abspath(save_path)} ===")
        
    except Exception as e:
        print(f"\nTraining interrupted by runtime exception: {str(e)}")

if __name__ == "__main__":
    main()

    