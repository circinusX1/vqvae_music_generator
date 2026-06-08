import yaml
import torch
import torch.nn as nn
import os
from src.data_loader import get_dataloader
from src.train_vqvae import VQVAEModel
from src.models.generator import MusicTransformer
from src.utils.audio_processing import setup_device

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
    
    # Initialize the standard data loader
    dataloader_obj = get_dataloader(
        cfg['dataset']['raw_dir'], cfg['training']['gen_batch_size'],
        cfg['dataset']['sample_rate'], cfg['dataset']['duration_sec']
    )
    
    if len(dataloader_obj) == 0:
        print("Error: No training batches generated. Check your data/raw folder.")
        return

    # Dynamically scale output layers to handle VQ-VAE codebook, the SOS token, and all config genres
    num_genres = len(cfg.get('genres', {}))
    vocab_extension_size = 1 + num_genres # 1 for SOS + N for your genre categories
    
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], cfg['generator']['num_layers'], cfg['generator']['num_heads']
    )
    
    # Reconfigure the linear head dynamically to match the absolute size of your expanded vocabulary
    transformer.fc_out = nn.Linear(cfg['generator']['embedding_dim'], cfg['generator']['num_embeddings'] + vocab_extension_size)
    transformer = transformer.to(device)
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg['training']['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # Vocab indexing mapping
    sos_token_id = cfg['generator']['num_embeddings']  # Index 512
    
    # VRAM Optimization Tools
    scaler = torch.amp.GradScaler('cuda') 
    accumulation_steps = 4               
    
    print(f"Beginning Multi-Genre Generation Prior Transformer Training ({cfg['training']['gen_epochs']} epochs)...")
    print(f"Detected genres in configuration: {list(cfg.get('genres', {}).keys())}")
    
    try:
        # Extract the underlying dataset file arrays to parse genre labels on-the-fly
        # Assumes dataset tracks paths internally via an accessible property like .audio_files or .filepaths
        dataset_ref = dataloader_obj.dataset
        has_file_tracking = hasattr(dataset_ref, 'audio_files') or hasattr(dataset_ref, 'filepaths')
        if not has_file_tracking:
            print("Warning: Dataset object does not contain a standard audio_files property list. Defaulting prefixes to first genre token.")

        for epoch in range(cfg['training']['gen_epochs']):
            total_loss = 0
            optimizer.zero_grad() 
            
            for batch_idx, batch in enumerate(dataloader_obj):
                batch_size_actual = batch.size(0)
                batch = batch.to(device)
                
                with torch.no_grad():
                    z = vqvae.encoder(batch)
                    _, _, indices = vqvae.quantizer(z) # Shape: [B, T_tokens]
                
                # --- GENRE CONDITIONING TRACK PREPARATION ---
                genre_token_list = []
                for b in range(batch_size_actual):
                    # Compute global file array item index relative to batch layout location
                    global_file_idx = (batch_idx * cfg['training']['gen_batch_size']) + b
                    
                    try:
                        if has_file_tracking:
                            file_list = dataset_ref.audio_files if hasattr(dataset_ref, 'audio_files') else dataset_ref.filepaths
                            track_path = file_list[global_file_idx % len(file_list)]
                            # Extract direct folder name (e.g. "data/raw/rock/riff.wav" -> "rock")
                            genre_name = os.path.basename(os.path.dirname(track_path))
                            genre_id = cfg['genres'].get(genre_name, sos_token_id + 1)
                        else:
                            genre_id = sos_token_id + 1
                    except Exception:
                        genre_id = sos_token_id + 1 # Dynamic fallback token protection
                        
                    genre_token_list.append(genre_id)
                
                genre_tensor = torch.tensor(genre_token_list, dtype=torch.long, device=device).unsqueeze(1) # Shape: [B, 1]
                sos_tokens = torch.full((batch_size_actual, 1), sos_token_id, dtype=torch.long, device=device) # Shape: [B, 1]
                
                # Build Prefix sequence architecture layout: [SOS, GENRE_TOKEN, AUDIO_TOKENS[:-2]]
                # Dropping the last 2 time steps keeps the exact temporal dimension expected by targets
                inputs = torch.cat([sos_tokens, genre_tensor, indices[:, :-2]], dim=1)
                targets = indices
                
                # Mixed Precision Forward Pass (Halves VRAM consumption for activations)
                with torch.amp.autocast('cuda'):
                    logits = transformer(inputs) # Output dimension spans entire vocabulary length cleanly
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / accumulation_steps
                
                # Mixed Precision Backward Pass
                scaler.scale(loss).backward()
                
                # Weight Update step happens every `accumulation_steps` iterations
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader_obj):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                total_loss += loss.item() * accumulation_steps
                
            print(f"Epoch {epoch+1}/{cfg['training']['gen_epochs']} - Average Loss: {total_loss/len(dataloader_obj):.4f}")
            
        # Explicitly save to root directory path cleanly
        save_path = cfg['training']['generator_path']
        torch.save(transformer.state_dict(), save_path)
        print(f"=== Success! Multi-Genre Prior Transformer weights successfully saved to: {os.path.abspath(save_path)} ===")
        
    except Exception as e:
        import traceback
        print(f"\nTraining interrupted by runtime exception: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    