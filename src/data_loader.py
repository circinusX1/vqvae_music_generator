import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

# In src/data_loader.py
class AudioDataset(Dataset):
    def __init__(self, raw_dir, sample_rate=22050, duration_sec=5):
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration_sec
        
        # Faster file discovery for 10k+ files
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
        #print(f" loading {file_path}")
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to Mono if Multi-channel
        if waveform.shape[0] > 1:
            print("??? should be mono already....")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or truncate to ensure uniform length [1, T]
        if waveform.shape[1] > self.target_length:
            max_start = waveform.shape[1] - self.target_length
            start = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start:start + self.target_length]
        else:
            # Pad if file is shorter than 5 seconds
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        original_samples = waveform.shape[1]
        used_samples = min(original_samples, self.target_length)
        utilization = (used_samples / original_samples) * 100
        
        print(f"File: {os.path.basename(file_path)} | Used: {used_samples/self.sample_rate:.2f}s out of {original_samples/sr:.2f}s ({utilization:.1f}%)")



        return waveform

def get_dataloader(raw_dir, batch_size, sample_rate, duration_sec, shuffle=True):
    dataset = AudioDataset(raw_dir, 
                           sample_rate, 
                           duration_sec)
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=2, 
                      pin_memory=True,
                      prefetch_factor=2) # Keep this low to avoid VRAM/RAM bloat
    
