import time
import itertools
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.utils import make_grid, save_image

from nets.generator import Generator
from nets.wgan_discriminator import Discriminator
from nets.classifier import Classifier
from nets.gan_discriminator import GANDiscriminator
from nets.resnet_feature_extractor import resnet18_feature_extractor

from utils.util_funcs import one_hot_embedding, weights_init_normal
from utils.argparser import arg_parser
from utils.datasets import ConcatDataset

# Get thar args from the user
opt = arg_parser()

# Check for cuda
cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2**4)
patch = (1, patch, patch)

# Loss function
adversarial_loss = torch.nn.BCELoss()
task_loss = torch.nn.CrossEntropyLoss()
dist_embed = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()

# Loss weights
lambda_adv = 1
lambda_task = 0.2
lambda_content_sim = 0.008

# Networks initialization
generator = Generator(opt)
discriminator = Discriminator(opt)
classifier = Classifier(opt)
gan_discriminator = GANDiscriminator(opt)

# Pretrained nets from Pytorch models
resnet_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_classifier.fc.in_features
for param in resnet_classifier.parameters():
    param.requires_grad = False
resnet_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_feature_extractor = resnet18_feature_extractor(pretrained=True)

generator.cuda()
discriminator.cuda()
classifier.cuda()
adversarial_loss.cuda()
task_loss.cuda()
resnet_classifier.cuda()
gan_discriminator.cuda()
resnet_feature_extractor.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)
gan_discriminator.apply(weights_init_normal)

# Visualization - TensorboardX ##############################################################################
writer = SummaryWriter(comment="_3_PCB_hard_fc")

writer.add_graph(generator,
                 (torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size).cuda(),
                  torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size).cuda(),
                  one_hot_embedding(torch.randint(opt.n_classes, (opt.batch_size,1)), opt.n_classes).cuda(),
                  torch.randn(opt.batch_size, opt.latent_dim).cuda()
                  )
                 )

writer.add_graph(discriminator,
                 (torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size).cuda(),
                  one_hot_embedding(torch.randint(opt.n_classes, (opt.batch_size,1)), opt.n_classes).cuda(),
                  torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size).cuda(),
                  torch.normal(torch.Tensor([0]), torch.Tensor([1])).cuda()
                  )
                 )
##############################################################################################################

# Data Loader ################################################################################################
data_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

syn_image_folder = datasets.ImageFolder(root='dataset/synthetic/rgb', transform=data_transform)
depth_image_folder = datasets.ImageFolder(root='dataset/synthetic/depth',
                                          transform=transforms.Compose([
                                             transforms.Resize((opt.img_size, opt.img_size)),
                                             transforms.ToTensor()
                                          ]))
ori_dataset = datasets.ImageFolder(root='dataset/test', transform=data_transform)

syn_dataset = ConcatDataset(syn_image_folder, depth_image_folder)

test_dataset = datasets.ImageFolder(root='dataset/test_2', transform=data_transform)

syn_loader = torch.utils.data.DataLoader(syn_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)

ori_loader = torch.utils.data.DataLoader(ori_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          drop_last=True)
###############################################################################################################

# Optimizers ##################################################################################################
optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(),
                                               classifier.parameters()),
                               lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_FD = torch.optim.Adam(gan_discriminator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_C = torch.optim.Adam(resnet_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
################################################################################################################

for epoch in range(opt.n_epochs):
    for i, (synth_images, synth_depth_images) in enumerate(syn_loader):
        batches_done = len(syn_loader) * epoch + i
        start_time = time.time()

        synthetic_images, synthetic_labels = synth_images
        synthetic_depth_imgs, synthetic_depth_labels = synth_depth_images

        real_images, real_labels = next(iter(ori_loader))
        test_images, test_labels = next(iter(test_loader))

        valid = torch.ones(opt.batch_size, *patch)
        fake = torch.zeros(opt.batch_size, *patch)
        valid_features = torch.zeros(opt.batch_size, 1)
        fake_features = torch.zeros(opt.batch_size, 1)

        # Sort the images by labels for viz purposes
        argsorted = torch.argsort(synthetic_labels)
        synthetic_images = synthetic_images[argsorted]
        synthetic_labels = synthetic_labels[argsorted]
        synthetic_depth_imgs = synthetic_depth_imgs[argsorted]

        real_argsorted = torch.argsort(real_labels)
        real_images = real_images[real_argsorted]
        real_labels = real_labels[real_argsorted]

        synthetic_images = synthetic_images.cuda()
        synthetic_labels = synthetic_labels.cuda()
        real_images = real_images.cuda()
        synthetic_depth_imgs = synthetic_depth_imgs.cuda()
        real_labels = real_labels.cuda()
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()
        valid = valid.cuda()
        valid_features = valid_features.cuda()
        fake = fake.cuda()
        fake_features = fake_features.cuda()

        foreground_mask = synthetic_depth_imgs.clone()
        m_foreground = foreground_mask <= 0
        m_background = foreground_mask >= 1
        foreground_mask[m_foreground] = 1
        foreground_mask[m_background] = 0
        synthetic_images[m_background] = 0

        onehot_syn = one_hot_embedding(synthetic_labels, opt.n_classes).cuda()
        onehot_real = one_hot_embedding(real_labels, opt.n_classes).cuda()

        # Generator and task training ############################################################
        optimizer_G.zero_grad()
        z = torch.FloatTensor(opt.batch_size, opt.latent_dim).uniform_(-1, 1).cuda()
        # z = Variable(FloatTensor(np.random.uniform(-1, 1, (opt.batch_size, opt.latent_dim))))
        # z = torch.Tensor(batch_size, opt.latent_dim).normal_(0, 2).cuda()
        std = torch.max(torch.FloatTensor([1]) - pow(epoch, 2)/opt.n_epochs, torch.FloatTensor([0]))
        disc_rand_noise = torch.randn(*synthetic_images.shape).cuda() * std.cuda()

        fake_images = generator(synthetic_images, synthetic_depth_imgs, onehot_syn, z)
        label_pred = classifier(fake_images)
        task_specific_loss = task_loss(label_pred, synthetic_labels)

        pixel_wise_sub = synthetic_images.view(-1) - fake_images.view(-1)
        fake_img_fore_patch = pixel_wise_sub * foreground_mask.view(-1)
        content_sim_loss = (1/opt.img_size*2) * torch.norm(fake_img_fore_patch) -\
                           (1/(opt.img_size*2)**2) * torch.dot(pixel_wise_sub, foreground_mask.view(-1))

        disc = discriminator(fake_images, onehot_syn, synthetic_depth_imgs, disc_rand_noise)
        wgan_generator_loss = -torch.mean(disc)

        synth_images_interest_patch = synthetic_images * foreground_mask
        fake_images_interest_patch = fake_images * foreground_mask

        synth_embeddings = resnet_feature_extractor(synth_images_interest_patch)
        fake_embeddings = resnet_feature_extractor(fake_images_interest_patch)

        feature_consistency_loss = l1_loss(synth_embeddings, fake_embeddings)

        if epoch % 5 == 0:
            gan_loss = adversarial_loss(gan_discriminator(fake_images, onehot_syn, disc_rand_noise), valid_features)
        else:
            gan_loss = 0

        generator_loss = lambda_task * task_specific_loss + \
                         lambda_adv * wgan_generator_loss + \
                         lambda_content_sim * content_sim_loss + \
                         0.17 * gan_loss + \
                         0.02 * feature_consistency_loss

        generator_loss.backward()
        optimizer_G.step()
        ##########################################################################################

        # Standalone Classifier ##################################################################
        optimizer_C.zero_grad()
        label_pred = resnet_classifier(fake_images.detach())
        eval_classifier_loss = task_loss(label_pred, synthetic_labels)
        eval_classifier_loss.backward()
        optimizer_C.step()
        ##########################################################################################

        # Discriminator training #################################################################
        wgan_discriminator_loss = None

        for _ in range(5):
            optimizer_D.zero_grad()

            shuff = torch.randperm(real_images.size(0))
            imgs_perm = real_images
            perm_hot = onehot_real
            fake_perm = fake_images
            fake_hot = onehot_syn
            synthetic_depth_imgs = synthetic_depth_imgs

            disc_real = discriminator(imgs_perm, perm_hot, synthetic_depth_imgs, disc_rand_noise)
            disc_fake = discriminator(fake_perm.detach(), fake_hot, synthetic_depth_imgs, disc_rand_noise)
            wgan_discriminator_loss = -(torch.mean(disc_real) - torch.mean(disc_fake))
            wgan_discriminator_loss.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.011, 0.011)
        ##########################################################################################

        # GAN Discriminator ##################################################################
        if epoch % 5 == 0:
            optimizer_FD.zero_grad()
            gan_discriminator_loss = adversarial_loss(gan_discriminator(real_images, onehot_real, disc_rand_noise), valid_features) + \
                                     adversarial_loss(gan_discriminator(fake_images.detach(), onehot_syn, disc_rand_noise), fake_features)

            gan_discriminator_loss.backward()
            optimizer_FD.step()
        ##########################################################################################

        # Evaluation metrics #####################################################################
        acc_synthetic = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == synthetic_labels.data.cpu().numpy())
        acc_real = np.mean(np.argmax(resnet_classifier(real_images).data.cpu().numpy(), axis=1) == real_labels.data.cpu().numpy())
        eval_real = np.mean(np.argmax(resnet_classifier(test_images).data.cpu().numpy(), axis=1) == test_labels.data.cpu().numpy())

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [FC loss: %f]"
              "[Synthetic acc: %3d%%, Real acc: %3d%%, Eval Real acc: %3d%%] [Time taken: %f Secs]" %
              (epoch, opt.n_epochs,
               i + 1, len(syn_loader),
               wgan_discriminator_loss.item(),
               generator_loss.item(),
               feature_consistency_loss.item(),
               100 * acc_synthetic,
               100 * acc_real,
               100 * eval_real,
               (time.time()-start_time) % 60))

        writer.add_scalar('Accuracy Synthetic', acc_synthetic, batches_done)
        writer.add_scalar('Accuracy Real', acc_real, batches_done)
        writer.add_scalar('Accuracy Real Evaluation', eval_real, batches_done)
        writer.add_scalar('Loss Content Similarity', content_sim_loss.item(), batches_done)
        writer.add_scalar('Loss Adversarial', wgan_generator_loss.item(), batches_done)
        writer.add_scalar('Loss Task Network', task_specific_loss.item(), batches_done)
        writer.add_scalar('Loss D', wgan_discriminator_loss.item(), batches_done)
        writer.add_scalar('Loss G', generator_loss.item(), batches_done)

        if batches_done % opt.sample_interval == 0:
            torch.save(generator.state_dict(), 'models_ckpt/3_PCB_hard_fc2.pt')
            sample = torch.cat((synthetic_images, fake_images, real_images))
            sample = make_grid(sample, normalize=True)
            writer.add_image("Images", sample, batches_done)
            # save_image(sample, 'images/%d.png' % batches_done, normalize=True)
