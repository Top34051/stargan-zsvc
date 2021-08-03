import numpy as np
import argparse
from utils import Audio

def sample_wav_audio(path):
    audio = Audio()
    mel = audio.audio_to_mel(path)
    samples = audio.mel_sample(mel, width=128, k=5)
    return samples

def save_embeddings(name, samples):
    audio = Audio()
    avg_embed = np.zeros(256, dtype=np.float32)
    for mel in samples:
        embed = audio.mel_to_embed(mel)
        avg_embed += embed
        avg_embed = avg_embed / 5
    np.save(f'./embeddings/{name}.npy', avg_embed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store', type=str, required=True)
    parser.add_argument('--name', action='store', type=str, required=True)
    args = parser.parse_args()

    samples = sample_wav_audio(args.path)
    save_embeddings(args.name, samples)