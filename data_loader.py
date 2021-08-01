import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader


class VCCDataset(Dataset):

    def __init__(self, path, batch_size):
        super().__init__()
        
        self.path = path
        self.batch_size = batch_size

        # load speaker embed
        self.embed = {}
        for fname in os.listdir('./embeddings'):
            if not fname.endswith('.npy'):
                continue
            spk = fname.split('.')[0]
            self.embed[spk] = np.load(os.path.join('./embeddings', fname))
        
        # load data
        self.data = []
        for fname in os.listdir(path):
            if not fname.endswith('.npy'):
                continue
            src = fname.split('.')[0]
            if src not in self.embed:
                continue

            data = np.load(os.path.join(path, fname))
            np.random.shuffle(data)
            
            # split to batches
            num = data.shape[0]
            batches = []
            i = 0
            while i + batch_size <= num:
                batches.append(data[i: i+batch_size])
                i += batch_size
            
            for batch in batches:
                trgs = random.choices(list(self.embed), k=3)
                for trg in trgs:
                    if src == trg:
                        continue
                    self.data.append((batch, self.embed[src], self.embed[trg]))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_dataloader(training=True, batch_size=16, num_workers=2):
    if training:
        dataset = VCCDataset('./data/train/', batch_size=batch_size)
    else:
        dataset = VCCDataset('./data/test/', batch_size=batch_size)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers
    )
    return data_loader