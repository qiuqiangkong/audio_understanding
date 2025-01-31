import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from piano_transcription_inference import PianoTranscription


class PianoTranscriptionCRnn(nn.Module):

    def __init__(self, sr: float, trainable: bool) -> None:
        r"""Piano transcription encoder [1]

        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, TASLP, 2022

        Code: https://github.com/qiuqiangkong/piano_transcription_inference
        """

        super().__init__()

        self.audio_sr = sr
        self.model_sr = 16000  # Piano transcription encoder sampling rate
        self.trainable = trainable

        self.model = PianoTranscription(device="cpu", checkpoint_path=None).model
        self.latent_dim = 88 * 4

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
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

        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

        # Forward
        with torch.set_grad_enabled(self.trainable):
            output_dict = self.model(audio)

        latent = torch.cat((
            output_dict["reg_onset_output"], 
            output_dict["reg_offset_output"], 
            output_dict["frame_output"], 
            output_dict["velocity_output"]
        ), dim=-1)  # shape: (b, t, d)

        return latent