import torch
from generator import MusicTransformer
from lora import merge_lora_weights
import yaml

cfg = yaml.safe_load(open("config/config.yaml"))

model = MusicTransformer(
    cfg['generator']['num_embeddings'],
    cfg['generator']['embedding_dim'],
    cfg['generator']['hidden_dim'],
    cfg['generator']['num_layers'],
    cfg['generator']['num_heads'],
    use_lora=True,
    lora_rank=cfg['generator'].get('lora_rank', 16),
    lora_alpha=cfg['generator'].get('lora_alpha', 32.0)
)

ckpt = torch.load(cfg['training']['generator_path'], map_location='cpu')
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=False)

merge_lora_weights(model)
torch.save(model.state_dict(), "generator.pt")   # clean version
print("Merged and saved clean generator.pt")
