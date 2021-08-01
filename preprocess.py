"""
Preprocess mel-spectrogram of size (80, 128) from VCC 2020 dataset
"""

import numpy as np
import os
import glob
import random
from sklearn.model_selection import train_test_split

from utils import Audio


def get_wav_paths(dirs):
    wav_paths = {}
    for dir in dirs:
        for spk in os.listdir(dir):
            if '.' in spk: continue
            wav_paths[spk] = glob.glob(os.path.join(dir, spk, '*.wav'))
    return wav_paths


def load_wav_audio(paths):
    data = {}
    mels = {}
    audio = Audio()
    for spk in paths:
        print('loading:', spk)
        data[spk] = []
        mels[spk] = []
        for path in paths[spk]:
            mel = audio.audio_to_mel(path)
            mels[spk].append(mel)
            samples = audio.mel_sample(mel, width=128, k=5)
            if samples is not None:
                data[spk].append(samples)
        data[spk] = np.concatenate(data[spk], axis=0)
    return data, mels


def create_dataset(data):
    os.makedirs('./data/train', exist_ok=True)
    os.makedirs('./data/test', exist_ok=True)
    for spk in data:
        train, test = train_test_split(data[spk], test_size=0.1, random_state=101)
        np.save(f'./data/train/{spk}.npy', train)
        np.save(f'./data/test/{spk}.npy', test)


def save_embeddings(mels):
    os.makedirs('./embeddings/', exist_ok=True)
    audio = Audio()
    for spk in mels:
        avg_embed = np.zeros(256, dtype=np.float32)
        for mel in mels[spk][:5]:
            embed = audio.mel_to_embed(mel)
            avg_embed += embed
        avg_embed = avg_embed / 5
        np.save(f'./embeddings/{spk}.npy', avg_embed)


if __name__ == "__main__":

    # 1. get data paths
    data_dirs = [
        '/home/jirayu.b/sing-me-light/data/source',
        '/home/jirayu.b/sing-me-light/data/target_task1',
        '/home/jirayu.b/sing-me-light/data/target_task2'
    ]
    paths = get_wav_paths(data_dirs)

    # 2. load wav audio
    data, mels = load_wav_audio(paths)

    # 3. create dataset
    create_dataset(data)

    # 4. save embeddings
    save_embeddings(mels)