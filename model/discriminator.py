import torch
from torch import nn
from fastai.layers import init_linear


class Discriminator(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed_dim = embed_dim
        self.input_dropout = nn.Dropout(p=0.3)

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )

        dims = [64, 128, 256, 512]
        block = []
        for i in range(1, len(dims)):
            cur, nxt = dims[i-1], dims[i]
            block.append(nn.Conv2d(in_channels=cur, out_channels=nxt*2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True))
            block.append(nn.InstanceNorm2d(num_features=nxt*2, affine=True))
            block.append(nn.GLU(dim=1))
        block.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=True))
        block.append(nn.InstanceNorm2d(num_features=1024, affine=True))
        block.append(nn.GLU(dim=1))
        self.down_sample = nn.Sequential(*block)

        self.linear = nn.Linear(in_features=512*10*16, out_features=1)

        self.embed_map_src = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.SELU()
        )
        self.embed_map_trg = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.SELU()
        )
        self.embed = nn.Linear(self.embed_dim*2, 512)
        init_linear(self.embed_map_src[0])
        init_linear(self.embed_map_trg[0])
        init_linear(self.embed)

    # x:   (bs, 80, 256)
    # src: (256,)
    # trg: (256,)
    def forward(self, x, src, trg, dropout=False):
        x = x.to(self.device)
        src = src.to(self.device)
        trg = trg.to(self.device)

        bs, _, width = x.shape

        src = self.embed_map_src(src)
        trg = self.embed_map_trg(trg)
        src_trg = torch.cat([src, trg], dim=1)
        embed = self.embed(src_trg)

        # input drop out
        if dropout:
            x = self.input_dropout(x)

        # init layer
        x = x.unsqueeze(1)
        x = self.init_layer(x)
        
        # down sampling layer
        x = self.down_sample(x)
        
        # global sum pooling
        h = torch.sum(x, dim=(-1, -2))
        x = self.linear(x.view(-1, 512*10*16))
        
        y = x + (embed[:, None]@h[..., None]).squeeze(-1)
        return y.view(bs)