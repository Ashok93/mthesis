import argparse


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00004, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--n_residual_blocks', type=int, default=6, help='number of residual blocks in generator')
    parser.add_argument('--latent_dim', type=int, default=64, help='dimensionality of the noise input')
    parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')
    parser.add_argument('--sample_interval', type=int, default=800, help='interval betwen image samples')

    opt = parser.parse_args()

    return opt
