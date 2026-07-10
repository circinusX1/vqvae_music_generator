import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, raw_dir, sample_rate=22050, duration_sec=1.5):
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration_sec)   # Ensure integer
        
        self.file_list = []
        valid_exts = {'.wav', '.mp3', '.flac'}
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    self.file_list.append(os.path.join(root, file))
            
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or truncate
        if waveform.shape[1] > self.target_length:
            max_start = waveform.shape[1] - self.target_length
            start = torch.randint(0, int(max_start), (1,)).item()   # Fixed: cast to int
            waveform = waveform[:, start:start + self.target_length]
        else:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

def get_dataloader(raw_dir, batch_size, sample_rate, duration_sec, shuffle=True):
    dataset = AudioDataset(raw_dir, sample_rate, duration_sec)
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=2, 
                      pin_memory=True,
                      prefetch_factor=2)
