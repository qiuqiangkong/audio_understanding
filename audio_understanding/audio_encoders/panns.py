from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
from panns_inference import AudioTagging


class PannsCnn14(nn.Module):

    def __init__(self, sr: float, trainable: bool) -> None:
        r"""Pretrained audio neural networks (PANNs) Cnn14 audio encoder [1]

        [1] Q. Kong, et al, PANNs: Large-scale pretrained audio neural networks 
        for audio pattern recognition, TASLP, 2020

        Code: https://github.com/qiuqiangkong/panns_inference
        """

        super().__init__()

        self.audio_sr = sr
        self.model_sr = 32000  # PANNs encoder sampling rate
        self.trainable = trainable

        self.model = AudioTagging(checkpoint_path=None, device="cpu").model
        self.latent_dim = 2048

    def encode(self, audio: torch.Tensor, train_mode: bool) -> torch.Tensor:
        r"""Extract audio latent.

        Args:
            audio: (b, c, t)

        Returns:
            latent: (b, t, d)
        """

        # Resample audio
        audio = torchaudio.functional.resample(
            waveform=audio, 
            orig_freq=self.audio_sr, 
            new_freq=self.model_sr
        )

        # To mono
        audio = torch.mean(audio, dim=1)  # shape: (b, t)

        if self.trainable and train_mode:
            self.model.train()
        else:
            self.model.eval()

        # Forward
        with torch.set_grad_enabled(self.trainable and train_mode):
            latent = self.model(audio)["embedding"]  # (b, d)
        
        latent = latent[:, None, :]
        # shape: (b, t, d)

        return latent