import torch
from torch import nn
from tqdm import tqdm

from model.discriminator import Discriminator
from model.generator import Generator


class Solver():

    def __init__(self, train_loader, test_loader, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = train_loader
        self.test_loader = test_loader

        # epoch
        self.epoch = 1

        # networks
        self.gen = Generator(embed_dim=256).to(self.device)
        self.dis = Discriminator(embed_dim=256).to(self.device)

        # train optimizers
        self.gen_lr = config['optimizers']['gen_lr']
        self.dis_lr = config['optimizers']['dis_lr']
        self.beta1 = config['optimizers']['beta1']
        self.beta2 = config['optimizers']['beta2']
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), self.gen_lr, [self.beta1, self.beta2])
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), self.dis_lr, [self.beta1, self.beta2])

        # hyperparams
        self.hparam = config['hparam']

        # load checkpoint
        if config['resume'] != '':
            checkpoint = torch.load(config['resume'])
            self.epoch = checkpoint['epoch'] + 1
            self.gen.load_state_dict(checkpoint['gen'])
            self.dis.load_state_dict(checkpoint['dis'])
        
        # losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def reset_grad(self):
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()

    def train_step(self, idx, x_src, src, trg):

        x_src = x_src.to(self.device)
        src = src.unsqueeze(0).to(self.device)
        trg = trg.unsqueeze(0).to(self.device)

        # inference
        x_src_src = self.gen(x_src, src, src)
        x_src_trg = self.gen(x_src, src, trg)
        x_src_trg_src = self.gen(x_src_trg, trg, src)
        d_src = self.dis(x_src, src, trg)
        d_src_trg = self.dis(x_src_trg, trg, src)

        # Train discriminator
        dis_loss = torch.mean((d_src_trg - self.hparam['b']) ** 2 + (d_src - self.hparam['a']) ** 2)

        self.reset_grad()
        dis_loss.backward(retain_graph=True)
        self.dis_opt.step()

        # Train generator
        if idx % 5 == 0:
            
            id_loss = self.l2_loss(x_src, x_src_src)
            cyc_loss = self.l1_loss(x_src, x_src_trg_src)

            d_src_trg_2 = self.dis(x_src_trg, trg, src)
            adv_loss = torch.mean((d_src_trg_2 - self.hparam['a']) ** 2)

            gen_loss = self.hparam['lambda_id'] * id_loss + self.hparam['lambda_cyc'] * cyc_loss + adv_loss

           
            self.reset_grad()
            gen_loss.backward(retain_graph=True)
            self.gen_opt.step()

            return dis_loss.item(), gen_loss.item()
        
        return dis_loss.item(), None

    def train(self, num_epoch=3000):

        # loop epoch
        while self.epoch <= num_epoch:

            print('Epoch {}'.format(self.epoch))

            gen_losses = []
            dis_losses = []

            # loop batch
            for idx, (mel, src, trg) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                gen_loss, dis_loss = self.train_step(idx+1, mel.squeeze(0), src.squeeze(0), trg.squeeze(0))
                
                gen_losses.append(gen_loss)
                if dis_loss is not None:
                    dis_losses.append(dis_loss)

            print('  gen loss: {}'.format(sum(gen_losses) / len(gen_losses)))
            print('  dis loss: {}'.format(sum(dis_losses) / len(dis_losses)))
            
            # save checkpoint
            torch.save({
                'epoch': self.epoch,
                'gen': self.gen.state_dict(),
                'dis': self.dis.state_dict()
            }, f'./checkpoints/checkpoint_{self.epoch}.pt')

            self.epoch += 1
