import yaml
import torch
import torch.nn.functional as F
import os
import gc
from data_loader import get_dataloader
from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device
from lora import count_parameters
import globals


def print_vram_usage(milestone_name, show_percent=True):
    if not torch.cuda.is_available():
        print(f"> {milestone_name}: CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        pct = (allocated / total) * 100 if total > 0 else 0
        pct_str = f" | {pct:.1f}% of {total:.0f}MB"
    except Exception:
        pct_str = ""
    print(f"> VRAM [{milestone_name}]  alloc: {allocated:7.1f}MB  reserved: {reserved:7.1f}MB  peak: {peak:7.1f}MB{pct_str}")


def sub_main(genere):
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = setup_device(cfg['training']['device'])
    print(f"\n{'='*60}")
    print(f"  Training Generator for genre: {genere}")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_lora = cfg['generator'].get('use_lora', True)
    lora_rank = cfg['generator'].get('lora_rank', 8)
    lora_alpha = cfg['generator'].get('lora_alpha', 16.0)
    lora_dropout = cfg['generator'].get('lora_dropout', 0.05)

    # Prefer best VQ-VAE
    vqvae_path = globals.vae_best_path(genere)
    if not os.path.exists(vqvae_path):
        vqvae_path = globals.vae_path(genere)
    print(f"\nLoading VQ-VAE from: {vqvae_path}")
    vqvae = VQVAEModel(cfg).to(device)
    if not os.path.exists(vqvae_path):
        print(f"Error: VQ-VAE not found at {vqvae_path}. Train VQ-VAE first.")
        return
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    print_vram_usage("After VQ-VAE loaded")

    train_duration = cfg['dataset'].get('duration_sec_train', 6.0)
    chunks_per_file = cfg['dataset'].get('train_chunks_per_file', 1)
    processed_dir = globals.gen_proc(genere)
    print(f"Loading data from: {processed_dir}")

    loader = get_dataloader(
        processed_dir,
        cfg['training']['gen_batch_size'],
        cfg['dataset']['sample_rate'],
        train_duration,
        chunks_per_file=chunks_per_file,
    )
    print(f"Dataset size: {len(loader.dataset)} samples")
    if len(loader) == 0:
        print(f"No data found in {processed_dir}!")
        return

    print("\nCreating MusicTransformer...")
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'],
        cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'],
        cfg['generator']['num_layers'],
        cfg['generator']['num_heads'],
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(transformer)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / max(total_params, 1):.2f}%)")
    if use_lora:
        print(f"LoRA enabled → rank={lora_rank}, alpha={lora_alpha}")
    print_vram_usage("After Transformer loaded")

    if use_lora:
        optimizer = torch.optim.Adam(transformer.get_trainable_params(), lr=cfg['training']['lr'])
    else:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg['training']['lr'])

    sos_token_id = cfg['generator']['num_embeddings']
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = cfg['generator'].get('accumulation_steps', 8)
    grad_clip = float(cfg['training'].get('grad_clip', 1.0))

    checkpoint_path = globals.gen_chk_path(genere)
    final_path = globals.gen_model_path(genere)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"\nResuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            try:
                transformer.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception as e:
                print(f"Warning while loading checkpoint: {e}")
            start_epoch = checkpoint.get('epoch', 0)
        else:
            try:
                transformer.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                print(f"Warning: could not load old-style checkpoint: {e}")

    print(f"\nStarting from epoch {start_epoch}...")
    print_vram_usage("Before training loop")

    plotter = globals.Ploter()
    plotter.ax.set_title(f'Live Generator Loss — {genere}')

    for epoch in range(start_epoch, cfg['training']['gen_epochs']):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            #wav_seconds += batch.shape[-1] / cfg['dataset']['sample_rate']
            batch = batch.to(device, non_blocking=True)

            with torch.no_grad():
                z = vqvae.encoder(batch)
                _, _, indices = vqvae.quantizer(z)

            max_ctx = cfg['generator'].get('max_context', 4096)
            if indices.size(1) > max_ctx:
                start = torch.randint(0, indices.size(1) - max_ctx + 1, (1,)).item()
                indices = indices[:, start:start + max_ctx]

            sos_tokens = torch.full((indices.size(0), 1), sos_token_id, dtype=torch.long, device=device)
            inputs = torch.cat([sos_tokens, indices[:, :-1]], dim=1)
            targets = indices

            with torch.amp.autocast('cuda'):
                logits = transformer(inputs)
                loss = F.kl_div(
                    F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1),
                    F.one_hot(targets.reshape(-1), num_classes=logits.size(-1)).float().to(device) * 0.9
                    + 0.1 / logits.size(-1),
                    reduction='batchmean',
                )
            loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                params_to_clip = transformer.get_trainable_params() if use_lora else transformer.parameters()
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accumulation_steps

            del batch, z, indices, sos_tokens, inputs, targets, logits
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        avg_loss = total_loss / max(len(loader), 1)
        plotter.add_point(avg_loss)
        print(f"\n[{genere}] Epoch {epoch+1}/{cfg['training']['gen_epochs']} - Avg Loss: {avg_loss:.4f}")
        print_vram_usage(f"End of Epoch {epoch+1}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'use_lora': use_lora,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'genere': genere,
        }, checkpoint_path)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n=== Generator training finished for genre: {genere} ===")

    if use_lora:
        print("Merging LoRA weights into base model...")
        transformer.merge_and_save(final_path)
    else:
        torch.save(transformer.state_dict(), final_path)

    print_vram_usage("After final merge & save")
    print(f"Final model saved to: {os.path.abspath(final_path)}")
    del plotter


def main():
    for genere, query in globals.GENERES.items():
        search_term, number_of_songs_to_download, enabled = query.split("|")
                
        if enabled == '0':
            print(f"\n>>> Skipping disabled genre: {genere} (Query: {query})")
            continue
        print(f"\n>>> Starting Generator training for genre: {genere} (Query: {query})")
        sub_main(genere)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

