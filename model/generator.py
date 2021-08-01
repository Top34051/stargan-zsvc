import torch
from torch import nn

from .condition_block import ConditioningBlock


class Generator(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # speaker embed
        self.embed_dim = embed_dim
        self.embed_map = nn.Sequential(
            nn.Linear(self.embed_dim*2, 256),
            nn.SELU()
        )

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7)),
            nn.GLU(dim=1)
        )

        dims = [64, 128, 256]
        block = []
        for i in range(1, len(dims)):
            cur, nxt = dims[i-1], dims[i]
            block.append(nn.Conv2d(in_channels=cur, out_channels=nxt*2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=True))
            block.append(nn.InstanceNorm2d(num_features=nxt*2, affine=True))
            block.append(nn.GLU(dim=1))
        self.down_sample = nn.Sequential(*block)

        self.down_converse = nn.Sequential(
            nn.Conv1d(in_channels=5120, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        self.cond_1 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_2 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_3 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_4 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_5 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_6 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_7 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_8 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)
        self.cond_9 = ConditioningBlock(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, embed_dim=self.embed_dim)

        self.up_converse = nn.Conv1d(in_channels=256, out_channels=5120, kernel_size=1, stride=1, padding=0, bias=True)

        self.up_sample = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(2), # channels / 4
            nn.GLU(dim=1),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(2),
            nn.GLU(dim=1),
        )

        self.out_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7), bias=True)


    # x:   (bs, 80, width)
    # src: (256,)
    # trg: (256,)
    def forward(self, x, src, trg):
        x = x.to(self.device)
        src = src.unsqueeze(0).to(self.device)
        trg = trg.unsqueeze(0).to(self.device)

        bs, _, width = x.shape
        src_trg = torch.cat([src, trg], dim=1)
        src_trg = self.embed_map(src_trg)

        # initialize layer
        x = x.unsqueeze(1)
        x = self.init_layer(x)
        
        # down sampling layer
        x = self.down_sample(x)
        
        # down conversion layer
        x = x.contiguous().view(bs, 5120, width // 4)

        x = self.down_converse(x)
        
        # bottleneck layer
        x = self.cond_1(x, src_trg)
        x = self.cond_2(x, src_trg)
        x = self.cond_3(x, src_trg)
        x = self.cond_4(x, src_trg)
        x = self.cond_5(x, src_trg)
        x = self.cond_6(x, src_trg)
        x = self.cond_7(x, src_trg)
        x = self.cond_8(x, src_trg)
        x = self.cond_9(x, src_trg)
        
        # up conversion layer
        x = self.up_converse(x)
        x = x.view(bs, 256, 20, width // 4)

        # up sampling layer
        x = self.up_sample(x)
        
        # output layer
        x = self.out_layer(x)
        
        return x.view(bs, 80, width)
