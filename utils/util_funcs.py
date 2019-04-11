import torch
import torch.functional as F


def compute_image_grad(images):
    x_filter = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).expand((1, 3, 3, 3)).cuda()
    y_filter = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).expand((1, 3, 3, 3)).cuda()
    images = images.cuda()
    g_x = F.conv2d(images, x_filter, padding=1)
    g_y = F.conv2d(images, y_filter, padding=1)

    grad = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))

    return grad


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]
