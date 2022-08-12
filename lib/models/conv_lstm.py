import torch
import torch.nn as nn
from torchvision import models, transforms

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class VideoModel(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=22, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.num_classes = num_classes

        #self.feat_extr = models.vgg16(pretrained=True)
        #self.feat_extr.classifier = self.feat_extr.classifier[:2]  # take output of fc6 layer
        #FEAT_VECT_DIM = self.feat_extr.classifier[0].out_features  # 4096
        FEAT_VECT_DIM = 4096

        #self.feat_extr = nn.Sequential(
        #    *list(self.feat_extr.children()),
        #    Flatten()
        #)
        #for param in self.feat_extr.parameters():
        #    param.requires_grad = False

        self.lin_transf = nn.Sequential(
            nn.Linear(FEAT_VECT_DIM, FEAT_VECT_DIM // 2),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTMCell(FEAT_VECT_DIM // 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x.shape == (batch_size, frames_per_sample, FEAT_VECT_DIM)
        h_n = torch.zeros(x.shape[0], self.hidden_dim, dtype=self.dtype, device=x.device)
        c_n = torch.zeros(x.shape[0], self.hidden_dim, dtype=self.dtype, device=x.device)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=self.dtype, device=x.device)
        for step in range(x.shape[1]):
            x_t = x[:, step]
            #out = self.feat_extr(x_t)
            out = self.lin_transf(x_t)
            h_n, c_n = self.lstm(out, (h_n, c_n))
            scores[:, step, :] = self.classifier(h_n)  # self.classifier(h_n).shape == (batch_size, num_classes)
        return scores