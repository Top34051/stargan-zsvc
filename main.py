from data_loader import get_dataloader
from utils import Audio
from model.generator import Generator
from model.discriminator import Discriminator
from data_loader import get_dataloader
from solver import Solver

import torch
from torch import nn
import numpy as np
from scipy.io.wavfile import write

if __name__ == "__main__":
    # audio = Audio()
    # mel = audio.audio_to_mel('/harddrive/harddrive10tb/top/vcc2020/source/SEF1/E10001.wav')
    # embed = audio.mel_to_embed(mel)
    # res = audio.mel_to_audio(mel)
    # print(mel.shape)
    # print(embed.shape)
    # print(res.shape)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # gen = Generator(embed_dim=256).to(device)
    # res = gen(
    #     torch.zeros(30, 80, 128), 
    #     torch.from_numpy(embed), 
    #     torch.from_numpy(embed)
    # )
    # print(res)

    # dis = Discriminator(embed_dim=256).to(device)
    # res = dis(
    #     torch.zeros(30, 80, 128), 
    #     torch.from_numpy(embed), 
    #     torch.from_numpy(embed)
    # )
    # print(res)

    # data = np.load('./data/train/TFF1.npy')
    # print(data.shape)

    train_loader = get_dataloader(training=True, batch_size=16, num_workers=2)
    test_loader = get_dataloader(training=False, batch_size=16, num_workers=2)

    solver = Solver(train_loader, test_loader, {
        'resume': '',
        'optimizers': {
            'gen_lr': 0.01,
            'dis_lr': 0.005,
            'beta1': 0.9,
            'beta2': 0.999
        },
        'hparam': {
            'a': 1,
            'b': 0,
            'lambda_id': 5,
            'lambda_cyc': 10
        }
    })
    solver.train(num_epoch=3000)

    # a = torch.zeros(16, 80, 128, requires_grad=True)
    # b = torch.zeros(16, 80, 128, requires_grad=True)

    # loss = nn.MSELoss()
    # out = loss(a, b)
    # print(out)
    # print(out.backward())