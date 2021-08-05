import torch
import numpy as np
import random

class Audio():
    """
    The class includes 
    1. Audio processing     : audio file to mel-spectrogram
    2. Speaker encoder      : mel-spectrogram to speaker identity vector
    3. Vocoder              : mel-spectrogram to audio file
    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        gru_embedder = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder')
        gru_embedder = gru_embedder.to(self.device)
        gru_embedder.eval()
        self.speaker_encoder = gru_embedder
        
        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(self.device)
        waveglow.eval()
        self.vocoder = waveglow

    # returns mel-spectrogram of size (80, T)
    def audio_to_mel(self, audio):
        mel = self.speaker_encoder.melspec_from_file(audio)
        mel = mel.transpose(-1, -2)
        return mel.data.cpu().numpy()

    # convert mel-spectrogram of size (80, T) to speaker embedding (256,)
    def mel_to_embed(self, mel):
        mel = np.expand_dims(mel.transpose(1, 0), axis=0)
        mel = torch.from_numpy(mel)
        mel = mel.to(self.device)
        embed = self.speaker_encoder(mel).squeeze(0)
        return embed.data.cpu().numpy()

    # converts mel-spectrogram to audio data
    def mel_to_audio(self, mel):
        mel = np.expand_dims(mel, axis=0)
        mel = torch.from_numpy(mel)
        mel = mel.to(self.device)
        with torch.no_grad():
            audio = self.vocoder.infer(mel)
        return audio[0].data.cpu().numpy()

    # sample k section from mel-spectrogram
    def mel_sample(self, mel, width=128, k=5):
        mel_width = mel.shape[1]
        if mel_width < width:
            return None
        pos = random.choices(range(mel_width - width), k=k)
        samples = np.array([mel[:, x: x+width] for x in pos])
        return samples