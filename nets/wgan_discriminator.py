import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.one_hot_fc = nn.Linear(opt.n_classes, opt.channels * opt.img_size ** 2)

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels*2, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.35, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.35, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.35, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.35, inplace=True),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img, onehot, depth_img, disc_rand_noise):
        dis_input = torch.cat((img, self.one_hot_fc(onehot).view(*img.shape)), 1)
        model_op = self.model(dis_input)
        return model_op
