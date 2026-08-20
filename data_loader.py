import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class AudioDataset(Dataset):
    """
    Each audio file contributes up to `chunks_per_file` training examples.
    Chunks are taken sequentially from the start of the file:
      chunk 0 → [0 : duration]
      chunk 1 → [duration : 2*duration]
      ...
    and are padded if a file is shorter than the required length.
    """
    def __init__(self, raw_dir, sample_rate=22050, duration_sec=1.5, chunks_per_file=1):
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration_sec)
        self.chunks_per_file = max(1, int(chunks_per_file))

        self.file_list = []
        valid_exts = {'.wav', '.mp3', '.flac'}
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    self.file_list.append(os.path.join(root, file))

        # RAM Cache dictionary to store loaded waveforms in memory
        self._cache = {}

    def __len__(self):
        # Expand dataset: each file yields up to chunks_per_file samples
        return len(self.file_list) * self.chunks_per_file

    def _load_mono(self, file_path):
        """Loads audio waveform into memory or retrieves it from RAM cache."""
        if file_path not in self._cache:
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            self._cache[file_path] = waveform

        return self._cache[file_path]

    def _sequential_crop_or_pad(self, waveform, chunk_idx):
        """Take the chunk_idx-th non-overlapping block starting from sample 0."""
        start = chunk_idx * self.target_length
        end = start + self.target_length
        total_len = waveform.shape[1]

        if start >= total_len:
            # File is too short for this chunk → return silence of target length
            return torch.zeros(1, self.target_length, dtype=waveform.dtype), start

        # Slice what is available
        chunk = waveform[:, start:min(end, total_len)]

        # Pad the remainder if the slice is shorter than target_length
        if chunk.shape[1] < self.target_length:
            padding = self.target_length - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))

        return chunk, start

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        chunk_idx = idx % self.chunks_per_file
        file_path = self.file_list[file_idx]
        waveform = self._load_mono(file_path)
        waveform, start = self._sequential_crop_or_pad(waveform, chunk_idx)
        return waveform


def get_dataloader(raw_dir, batch_size, sample_rate, duration_sec,
                   chunks_per_file=1, shuffle=True):
    dataset = AudioDataset(
        raw_dir,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        chunks_per_file=chunks_per_file,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )
