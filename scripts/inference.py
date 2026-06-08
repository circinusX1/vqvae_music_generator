import yaml
import torch
import torchaudio
from src.train_vqvae import VQVAEModel
from src.models.generator import MusicTransformer
from src.utils.audio_processing import setup_device

def generate_long_music(output_path="output_15s.wav", target_duration_sec=15):
    with open("config/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = setup_device(cfg['training']['device'])
    
    # 1. Calculate precise token requirements based on architecture
    sample_rate = cfg['dataset']['sample_rate']
    stride = cfg['vqvae']['stride']
    tokens_per_second = sample_rate / stride
    total_tokens_needed = int(target_duration_sec * tokens_per_second)
    
    # Context window constraint matching training limits (5 seconds)
    max_context_window = int(cfg['dataset']['duration_sec'] * tokens_per_second)
    
    print(f"Targeting {target_duration_sec}s of audio.")
    print(f"Generating {total_tokens_needed} tokens total using a sliding context window of {max_context_window} tokens.")

    # Load Models
    vqvae = VQVAEModel(cfg).to(device)
    vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))
    vqvae.eval()
    
    transformer = MusicTransformer(
        cfg['generator']['num_embeddings'], cfg['generator']['embedding_dim'],
        cfg['generator']['hidden_dim'], cfg['generator']['num_layers'], cfg['generator']['num_heads']
    ).to(device)
    transformer.load_state_dict(torch.load(cfg['training']['generator_path'], map_location=device))
    transformer.eval()
    
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
            
            # --- TEMPERATURE SAMPLING FIX ---
            # Instead of argmax (which creates robotic loops), we add soft scaling
            temperature = 0.85
            filtered_logits = next_token_logits / temperature
            
            # Force the model to avoid selecting the SOS/Padding token again during generation
            filtered_logits[:, sos_token_id] = -float('Inf')
            
            probabilities = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
        if (i + 1) % 5000 == 0:
            print(f"Generated {i + 1}/{total_tokens_needed} tokens... (Last Token Sampled: {next_token.item()})")
            
    # Strip away initialization SOS token
    final_indices = generated_sequence[:, 1:]
    
    print(f"Token map generation complete. Array length: {final_indices.shape[1]} entries.")
    print(f"Unique tokens utilized in this sequence: {len(torch.unique(final_indices))}/{cfg['vqvae']['num_embeddings']}")
    
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

    