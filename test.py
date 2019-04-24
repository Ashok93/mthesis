import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms

from torchvision import datasets, models
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

from nets.generator import Generator
from nets.classifier import Classifier

from utils.util_funcs import one_hot_embedding, weights_init_normal
from utils.argparser import arg_parser
from utils.datasets import ConcatDataset

writer = SummaryWriter(comment="_evaluation_pcb_only")

# Get thar args from the user
opt = arg_parser()

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2**4)
patch = (1, patch, patch)

cuda = True if torch.cuda.is_available() else False

# Loss function
task_loss = torch.nn.CrossEntropyLoss()

# Loss weights
lambda_adv = 1
lambda_task = 0.3
lambda_content_sim = 0.0002

# Initialize generator and discriminator
generator = Generator(opt)
classifier = Classifier(opt)

# Pretrained Resnet Classifiers for different training sets ############################################################
resnet_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_classifier.fc.in_features
for param in resnet_classifier.parameters():
    param.requires_grad = False
resnet_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_source_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_source_classifier.fc.in_features
for param in resnet_source_classifier.parameters():
    param.requires_grad = False
resnet_source_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_target_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_target_classifier.fc.in_features
for param in resnet_target_classifier.parameters():
    param.requires_grad = False
resnet_target_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_target_source_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_target_source_classifier.fc.in_features
for param in resnet_target_source_classifier.parameters():
    param.requires_grad = False
resnet_target_source_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_target_fake_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_target_fake_classifier.fc.in_features
for param in resnet_target_fake_classifier.parameters():
    param.requires_grad = False
resnet_target_fake_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_classifier = resnet_classifier.cuda()
resnet_source_classifier = resnet_source_classifier.cuda()
resnet_target_classifier = resnet_target_classifier.cuda()
resnet_target_source_classifier = resnet_target_source_classifier.cuda()
resnet_target_fake_classifier = resnet_target_fake_classifier.cuda()
########################################################################################################################

if cuda:
    generator.cuda()
    classifier.cuda()
    task_loss.cuda()

# Initialize weights
generator.load_state_dict(torch.load('models_ckpt/pcb_only.pt'))
# generator.train()
classifier.apply(weights_init_normal)

data_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

syn_image_folder = datasets.ImageFolder(root='dataset/synthetic/rgb',
                                        transform=data_transform)
depth_image_folder = datasets.ImageFolder(root='dataset/synthetic/depth',
                                         transform=transforms.Compose([
                                             transforms.Resize((opt.img_size, opt.img_size)),
                                             transforms.ToTensor()
                                         ]))
syn_dataset = ConcatDataset(syn_image_folder, depth_image_folder)
ori_dataset = datasets.ImageFolder(root='dataset/test',
                                   transform=data_transform)

test_dataset = datasets.ImageFolder(root='dataset/test_2',
                                   transform=data_transform)

# DataLoader for the datasets
syn_loader = torch.utils.data.DataLoader(syn_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)

ori_loader = torch.utils.data.DataLoader(ori_dataset,
                                         batch_size=opt.batch_size, shuffle=True,
                                         num_workers=2, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=30, shuffle=True,
                                         num_workers=2, drop_last=True)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Optimizers
optimizer_C = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_fake = torch.optim.Adam(resnet_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_source = torch.optim.Adam(resnet_source_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_target = torch.optim.Adam(resnet_target_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_target_fake = torch.optim.Adam(resnet_target_fake_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_target_source = torch.optim.Adam(resnet_target_source_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

confusion_matrix = torch.zeros(opt.n_classes, opt.n_classes)

for epoch in range(opt.n_epochs):
    for i, ((synth_images, synth_depth_images), (real_images, real_labels)) in enumerate(zip(syn_loader, ori_loader)):
        start_time = time.time()
        batches_done = len(syn_loader) * epoch + i

        synthetic_images, synthetic_labels = synth_images
        synthetic_depth_imgs, synthetic_depth_labels = synth_depth_images
        test_images, test_labels = next(iter(test_loader))

        batch_size = synthetic_images.size(0)
        batch_size_m = real_images.size(0)
        valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        valid_m = Variable(FloatTensor(batch_size_m, *patch).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

        # Configure input
        synthetic_images = Variable(synthetic_images.type(FloatTensor))
        synthetic_labels = Variable(synthetic_labels.type(LongTensor))
        real_images = Variable(real_images.type(FloatTensor))
        synthetic_depth_imgs = Variable(synthetic_depth_imgs.type(FloatTensor))
        real_labels = Variable(real_labels.type(LongTensor))
        test_images = Variable(test_images.type(FloatTensor))
        test_labels = Variable(test_labels.type(LongTensor))

        foreground_mask = synthetic_depth_imgs.clone()
        m_foreground = foreground_mask <= 0
        m_background = foreground_mask >= 1

        foreground_mask[m_foreground] = 1
        foreground_mask[m_background] = 0
        synthetic_images[m_background] = 0

        onehot_syn = one_hot_embedding(synthetic_labels, opt.n_classes).cuda()
        onehot_real = one_hot_embedding(real_labels, opt.n_classes).cuda()

        # Generator and task training ############################################################
        optimizer_fake.zero_grad()
        z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))))
        fake_images = generator(synthetic_images, synthetic_depth_imgs, onehot_syn, z)
        label_pred = resnet_classifier(fake_images.detach())
        eval_classifier_loss = task_loss(label_pred, synthetic_labels)
        eval_classifier_loss.backward()
        optimizer_fake.step()

        optimizer_source.zero_grad()
        source_label_pred = resnet_source_classifier(synthetic_images)
        source_only_loss = task_loss(source_label_pred, synthetic_labels)
        source_only_loss.backward()
        optimizer_source.step()

        optimizer_target.zero_grad()
        target_label_pred = resnet_target_classifier(real_images)
        target_only_loss = task_loss(target_label_pred, real_labels)
        target_only_loss.backward()
        optimizer_target.step()

        optimizer_target_fake.zero_grad()
        target_fake_images = torch.cat((fake_images[0:int(opt.batch_size/2)-1].detach(),
                                       real_images[int(opt.batch_size/2):opt.batch_size-1]))
        target_label_pred = resnet_target_fake_classifier(target_fake_images)
        target_fake_label = torch.cat((synthetic_labels[0:int(opt.batch_size/2)-1],
                                       real_labels[int(opt.batch_size/2):opt.batch_size-1]))
        target_only_loss = task_loss(target_label_pred, target_fake_label)
        target_only_loss.backward()
        optimizer_target_fake.step()

        optimizer_target_source.zero_grad()
        target_source_images = torch.cat((synthetic_images[0:int(opt.batch_size / 2) - 1],
                                         real_images[int(opt.batch_size / 2):opt.batch_size - 1]))
        target_label_pred = resnet_target_source_classifier(target_source_images)
        target_source_label = torch.cat((synthetic_labels[0:int(opt.batch_size/2)-1],
                                         real_labels[int(opt.batch_size/2):opt.batch_size-1]))
        target_only_loss = task_loss(target_label_pred, target_source_label)
        target_only_loss.backward()
        optimizer_target_source.step()

        ##########################################################################################

        # Evaluation metrics #####################################################################
        train_pred = resnet_classifier(test_images)
        source_pred = resnet_source_classifier(test_images)
        target_pred = resnet_target_classifier(test_images)
        target_fake_pred = resnet_target_fake_classifier(test_images)
        target_source_pred = resnet_target_source_classifier(test_images)

        train_pred = np.argmax(train_pred.data.cpu().numpy(), axis=1)
        train_label = test_labels.data.cpu().numpy()
        source_pred = np.argmax(source_pred.data.cpu().numpy(), axis=1)
        target_pred = np.argmax(target_pred.data.cpu().numpy(), axis=1)
        target_fake_pred = np.argmax(target_fake_pred.data.cpu().numpy(), axis=1)
        target_source_pred = np.argmax(target_source_pred.data.cpu().numpy(), axis=1)

        acc_synthetic = np.mean(train_pred == train_label)
        acc_source = np.mean(source_pred == train_label)
        acc_target = np.mean(target_pred == train_label)
        acc_target_fake = np.mean(target_fake_pred == train_label)
        acc_target_source = np.mean(target_source_pred == train_label)
        confusion_matrix[train_pred, train_label] += 1

        print("[Epoch %d/%d] [Batch %d/%d] [C loss: %f] [Fake acc: %3d%%] [Source Only acc: %3d%%]"
              " [Target Only acc: %3d%%] [Fake + Target Only acc: %3d%%] [Fake + Source Only acc: %3d%%]"
              " [Time taken: %f Secs]" %
              (epoch, opt.n_epochs,
               i+1, len(ori_loader),
               eval_classifier_loss.item(),
               100*acc_synthetic,
               100*acc_source,
               100*acc_target,
               100*acc_target_fake,
               100*acc_target_source,
               (time.time()-start_time) % 60))

        writer.add_scalar('Accuracy Fake', acc_synthetic, batches_done)
        writer.add_scalar('Accuracy Source Only', acc_source, batches_done)
        writer.add_scalar('Accuracy Target Only', acc_target, batches_done)
        writer.add_scalar('Accuracy Target + Fake', acc_target_fake, batches_done)
        writer.add_scalar('Accuracy Target + Source', acc_target_source, batches_done)

        # print(confusion_matrix)

        batches_done = len(syn_loader) * epoch + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((synthetic_images, fake_images, real_images))
            sample = make_grid(sample, normalize=True)
            writer.add_image("Images", sample, batches_done)
            save_image(sample, 'images/%d.png' % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)
