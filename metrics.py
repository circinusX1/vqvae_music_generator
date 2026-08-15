import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale STFT loss (EnCodec / SoundStream / HiFi-GAN style).
    Runs in float32 for numerical stability under AMP.
    """

    def __init__(
        self,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=None,
        win_lengths=None,
        mag_weight=1.0,
        sc_weight=1.0,
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = hop_sizes or [s // 4 for s in self.fft_sizes]
        self.win_lengths = win_lengths or list(self.fft_sizes)
        self.mag_weight = mag_weight
        self.sc_weight = sc_weight

        for i, win in enumerate(self.win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(win), persistent=False)

    def _stft(self, x, fft_size, hop_size, win_length, window):
        if x.dim() == 3:
            x = x.squeeze(1)
        window = window.to(device=x.device, dtype=x.dtype)
        return torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

    def forward(self, target, reconstruction):
        # Force float32 – STFT + log is unstable in fp16
        target = target.float()
        reconstruction = reconstruction.float()

        min_len = min(target.shape[-1], reconstruction.shape[-1])
        target = target[..., :min_len]
        reconstruction = reconstruction[..., :min_len]

        total_loss = target.new_zeros(())
        for i, (fft, hop, win) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            window = getattr(self, f"window_{i}")
            stft_t = self._stft(target, fft, hop, win, window)
            stft_r = self._stft(reconstruction, fft, hop, win, window)

            mag_t = torch.abs(stft_t).clamp_min(1e-5)
            mag_r = torch.abs(stft_r).clamp_min(1e-5)

            sc_loss = torch.norm(mag_t - mag_r, p="fro") / (torch.norm(mag_t, p="fro") + 1e-5)
            mag_loss = F.l1_loss(torch.log(mag_r), torch.log(mag_t))

            total_loss = total_loss + self.sc_weight * sc_loss + self.mag_weight * mag_loss

        wave_loss = F.l1_loss(reconstruction, target)
        total_loss = total_loss + 0.1 * wave_loss

        loss = total_loss / len(self.fft_sizes)

        # Guard against NaN/Inf
        if not torch.isfinite(loss):
            loss = wave_loss.detach()

        return loss

