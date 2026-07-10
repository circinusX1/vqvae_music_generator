# inference_minimal.py
import yaml, torch, torchaudio, os
from train_vqvae import VQVAEModel
from generator import MusicTransformer
from audio_processing import setup_device

cfg = yaml.safe_load(open("config/config.yaml"))
device = setup_device(cfg['training']['device'])

vqvae = VQVAEModel(cfg).to(device).eval()
vqvae.load_state_dict(torch.load(cfg['training']['vqvae_path'], map_location=device))

trans = MusicTransformer(cfg['generator']['num_embeddings'], 
                         cfg['generator']['embedding_dim'],
                         cfg['generator']['hidden_dim'], 
                         cfg['generator']['num_layers'], 
                         cfg['generator']['num_heads']).to(device).eval()
trans.load_state_dict(torch.load(cfg['training']['generator_path'], map_location=device))

sos = cfg['generator']['num_embeddings']
seq = torch.full((1,1), sos, dtype=torch.long, device=device)
total = int(8 * cfg['dataset']['sample_rate'] / cfg['vqvae']['stride'])   # 8 seconds

for i in range(total):
    ctx = seq[:, -384:]
    logits = trans(ctx)[:, -1, :] / 1.1          # high temperature
    next_t = torch.multinomial(torch.softmax(logits, -1), 1)
    seq = torch.cat([seq, next_t], 1)

# decode
indices = seq[:,1:]
chunks = []
for s in range(0, indices.shape[1], 1024):
    z = vqvae.quantizer.embedding(indices[:,s:s+1024]).permute(0,2,1)
    chunks.append(vqvae.decoder(z).cpu())

torchaudio.save("test_output.wav", torch.cat(chunks, -1).squeeze(0), cfg['dataset']['sample_rate'])
print("Saved test_output.wav")
