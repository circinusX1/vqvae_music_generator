import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, raw_dir, sample_rate=22050, duration_sec=5):
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration_sec
        # Find all common audio extensions
        self.file_list = []
        for ext in ('*.wav', '*.mp3', '*.flac', '*.WAV', '*.MP3', '*.FLAC'):
            self.file_list.extend(glob.glob(os.path.join(raw_dir, ext)))
            
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to Mono if Multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or truncate to ensure uniform length [1, T]
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform

def get_dataloader(raw_dir, batch_size, sample_rate, duration_sec, shuffle=True):
    dataset = AudioDataset(raw_dir, sample_rate, duration_sec)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    
