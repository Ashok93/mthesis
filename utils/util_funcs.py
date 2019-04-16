import torch
import torch.functional as F
import numpy as np


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


def compute_gradient_penalty(D, one_hot, depth_images, disc_rand_noise, patch, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, one_hot, depth_images, disc_rand_noise)
    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], *patch).fill_(1.0), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean().cuda()
    return gradient_penalty
