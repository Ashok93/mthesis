import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ PixelDA Generator Model
    Generator conditioned on noise z as well as input image
    Network = n64s1 --> ReLu --> Residual_block(as in paper) --> n3s1 --> tanh --> img
    Noise z --> FC --> Network
    Image --> Network
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.fc = nn.Linear(opt.latent_dim, opt.channels*opt.img_size**2)
        self.one_hot_fc = nn.Linear(opt.n_classes, opt.channels * opt.img_size ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(opt.channels*4, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(opt.n_residual_blocks)])
        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())
        self.model = nn.Sequential(
            self.l1,
            self.res_blocks,
            self.l2
        )

    def forward(self, img, depth_img, onehot_syn, z):
        img_plus_depth = torch.cat((img, depth_img.expand(*img.shape)), 1)
        gen_input = torch.cat((img_plus_depth, self.fc(z).view(*img.shape)), 1)
        gen_input = torch.cat((gen_input, self.one_hot_fc(onehot_syn).view(*img.shape)), 1)
        gen_images = self.model(gen_input)
        return gen_images


class SegmentationMapGenerator(Generator):

    def __init__(self, opt):
        super(SegmentationMapGenerator, self).__init__(opt)
        self.l1 = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Sigmoid())

        self.model = nn.Sequential(
            self.l1,
            self.res_blocks,
            self.l2
        )

    def forward(self, img):
        gen_image = self.model(img)
        return gen_image
