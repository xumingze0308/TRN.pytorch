import _init_paths
import utils as utl

import os
import numpy as np

from torchvision import models, transforms
import torch
import torch.nn as nn

from PIL import Image

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(pretrained=True)
    model = nn.Sequential(
        *list(model.children())[:-1],
        Flatten(),     # feat_vect_dim: 512*7*7
    ).to(device)
    model.train(False)
    FEAT_VECT_DIM = 512*7*7

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    SAMPLE_FRAMES = 6     # take only the central frame every six frames

    DATA_ROOT = 'data/THUMOS'

    VIDEO_FRAMES = 'video_frames_24fps'   # base folder where the video folders (containing the frames) are
    TARGET_FRAMES = 'target_frames_24fps' # labels for the frames above

    VIDEO_FEATURES = 'resnet200-fc'
    OPTIC_FEATURES = 'bn_inception'
    TARGET_FEATURES = 'target'


    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if 'video' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % SAMPLE_FRAMES)

            frames = torch.zeros(num_frames//SAMPLE_FRAMES, FEAT_VECT_DIM)
            junk = torch.zeros(num_frames//SAMPLE_FRAMES, 1024)  # optical flow will not be used
            count = 0
            for idx_frame in range(SAMPLE_FRAMES//2, num_frames, SAMPLE_FRAMES):
                # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                frame = Image.open(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir, str(idx_frame+1)+'.jpg')).convert('RGB')
                frame = transform(frame).to(device)
                # forward pass
                feat_vect = model(frame.unsqueeze(0))
                frames[count] = feat_vect.squeeze(0)

                count += 1

            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)+'.npy'), frames.numpy())
            np.save(os.path.join(DATA_ROOT, OPTIC_FEATURES, str(dir) +'.npy'), junk.numpy())
            target = np.load(os.path.join(DATA_ROOT, TARGET_FRAMES, dir+'.npy'))[:num_frames]
            target = target[SAMPLE_FRAMES//2::SAMPLE_FRAMES]
            np.save(os.path.join(DATA_ROOT, TARGET_FEATURES, str(dir)+'.npy'), target)

if __name__ == '__main__':
    main()