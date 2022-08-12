import os

import torch
import torch.nn as nn
from torch import optim

import _init_paths
import utils as utl
from configs.thumos import parse_trn_args as parse_args
from lib.models.conv_lstm import VideoModel

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))

    model = VideoModel(args.hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        data_loaders = {
            phase: utl.build_data_loader(args, phase)
            for phase in args.phases
        }

        for phase in args.phases:
            training = phase=='train'
            if training:
                model.train(True)
            elif not training and args.debug:
                model.train(False)
            else:
                continue

            with torch.set_grad_enabled(training):
                avg_loss = 0
                for batch_idx, (camera_inputs, motion_inputs, enc_target, dec_target) \
                        in enumerate(data_loaders[phase], start=1):
                    camera_inputs = camera_inputs.to(device)
                    enc_target = enc_target.to(device)

                    if training:
                        optimizer.zero_grad()

                    scores = model(camera_inputs)

                    # sum losses along all timesteps
                    loss = criterion(scores[:, 0], enc_target[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss += criterion(scores[:, step], enc_target[:, step].max(axis=1)[1])

                    if training:
                        loss.backward()
                        optimizer.step()

                    avg_loss += loss.item()
                    print('{:5s} Epoch:{}  Iteration:{}  Loss:{:.3f}'.format(phase, epoch + 1, batch_idx, loss.item()))

                print('-- {:5s} Epoch:{} avg_loss:{:.3f}'.format(phase, epoch + 1, avg_loss / batch_idx))

if __name__ == '__main__':
    main(parse_args())