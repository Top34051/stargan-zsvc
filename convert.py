import argparse
import numpy as np
import torch
import soundfile as sf

from utils import Audio
from model.generator import Generator


INTERVAL = 128

def convert_audio(audio_path, src_id, trg_id, out_path, checkpoint='./checkpoints/best.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio = Audio()
    gen = Generator(embed_dim=256, device=device).to(device)

    checkpoint = torch.load(checkpoint)
    gen.load_state_dict(checkpoint['gen'])

    src = np.load(f'./embeddings/{src_id}.npy')
    trg = np.load(f'./embeddings/{trg_id}.npy')

    mel = audio.audio_to_mel(audio_path)
    n = mel.shape[1]

    cur = 0
    res = []
    
    while cur + INTERVAL < n:
        mel_res = gen(
            torch.from_numpy(np.expand_dims(mel[:, cur: cur+INTERVAL], 0)), 
            torch.from_numpy(np.expand_dims(src, 0)), 
            torch.from_numpy(np.expand_dims(trg, 0))
        )
        mel_res = mel_res[0].data.cpu().numpy()
        res = np.concatenate((res, audio.mel_to_audio(mel_res)), axis=None)
        cur += INTERVAL

    sf.write(out_path, res, samplerate=22050)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', action='store', type=str, required=True)
    parser.add_argument('--src_id', action='store', type=str, required=True)
    parser.add_argument('--trg_id', action='store', type=str, required=True)
    parser.add_argument('--out_path', action='store', type=str, default='./results/out.wav')
    args = parser.parse_args()

    convert_audio(args.audio_path, args.src_id, args.trg_id, args.out_path)
