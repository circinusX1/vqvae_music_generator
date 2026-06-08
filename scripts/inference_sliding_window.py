import yaml
import torch
import torchaudio
from src.train_vqvae import VQVAEModel
from src.models.generator import MusicTransformer
from src.utils.audio_processing import setup_device


def print_vram_usage(milestone_name):
    """Helper to print current and max VRAM utilization in Megabytes"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"> {milestone_name}: alocated: {allocated:.2f}MB, reserved {reserved:.2f}MB,  peak: {max_allocated:.2f}MB\n")


def generate_long_music(output_path="output_15s_slw.wav", target_duration_sec=15):
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = setup_device(cfg['training']['device'])
    
    # 1. Calculate precise token requirements based on architecture architecture
    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    tokens_per_second = sample_rate / stride
    total_tokens_needed = int(target_duration_sec * tokens_per_second)
    
    # Context window constraint matching training limits (5 seconds)
    max_context_window = int(cfg['dataset']['duration_sec'] * tokens_per_second)
    
    print(f"Targeting {target_duration_sec}s of audio.")
    print(f"Generating {total_tokens_needed} tokens total using a sliding context window of {max_context_window} tokens.")


    print_vram_usage("before Q-VAE Weights Loaded")

    # Load Models
    vqvae = VQVAEModel(cfg).to(device)
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae.eval()

    print_vram_usage("VQ-VAE Weights Loaded")
    
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], cfg['generator']['num_layers'], cfg['generator']['num_heads']
    ).to(device)
    transformer.load_state_dict(torch.load(cfg['training']['generator_path'], map_location=device))
    transformer.eval()

    print_vram_usage("VQ-VAE Transformer Loaded")
    
    sos_token_id = cfg['generator']['num_embeddings']
    generated_sequence = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
    
    print("Autoregressively generating codebook track sequence...")
    
    # 2. Sliding window generation loop
    for i in range(total_tokens_needed):
        # Truncate oldest history context if sequence exceeds training capacity
        if generated_sequence.size(1) > max_context_window:
            context = generated_sequence[:, -max_context_window:]
        else:
            context = generated_sequence
            
        with torch.no_grad():
            logits = transformer(context)
            next_token_logits = logits[:, -1, :]
            
            # Optional: Add temperature sampling to prevent repetitive looping artifacts
            # probabilities = torch.softmax(next_token_logits / 0.9, dim=-1)
            # next_token = torch.multinomial(probabilities, num_samples=1)
            
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
        if (i + 1) % 1000 == 0:
            print_vram_usage(f"token: {i} ")
            perc = (i * 100)/total_tokens_needed
            print(f"Generated {i} -> {perc:.2f}% tokens...")
            
    # Strip away initialization SOS token
    final_indices = generated_sequence[:, 1:]
    
    print("Decoding discrete token map into raw continuous audio waveform...")
    with torch.no_grad():
        quantized_vectors = vqvae.quantizer.embedding(final_indices)
        quantized_vectors = quantized_vectors.permute(0, 2, 1).contiguous()
        generated_waveform = vqvae.decoder(quantized_vectors)
        
    # Export back into raw wave system storage
    torchaudio.save(output_path, generated_waveform.cpu().squeeze(0), sample_rate)
    print(f"File successfully rendered to {output_path}")

if __name__ == "__main__":
    generate_long_music()
    