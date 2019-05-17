import torch
import torch.nn as nn


class GANDiscriminator(nn.Module):

    def __init__(self, opt):
        super(GANDiscriminator, self).__init__()
        self.one_hot_fc = nn.Linear(opt.n_classes, opt.channels * opt.img_size ** 2)

        ndf = 64
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.channels*2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, onehot, disc_rand_noise):
        img = img + disc_rand_noise
        dis_input = torch.cat((img, self.one_hot_fc(onehot).view(*img.shape)), 1)
        model_op = self.model(dis_input)
        return model_op
