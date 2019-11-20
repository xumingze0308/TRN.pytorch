import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from datasets import build_dataset

__all__ = [
    'set_seed',
    'build_data_loader',
    'weights_init',
    'count_parameters',
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_data_loader(args, phase='train'):
    data_loaders = data.DataLoader(
        dataset=build_dataset(args, phase),
        batch_size=args.batch_size,
        shuffle=phase=='train',
        num_workers=args.num_workers,
    )
    return data_loaders

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
