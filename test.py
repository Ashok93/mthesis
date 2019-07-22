import time
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from torchvision import datasets, models
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from nets.generator import Generator, SegmentationMapGenerator
from nets.classifier import Classifier

from utils.util_funcs import one_hot_embedding, weights_init_normal
from utils.argparser import arg_parser
from utils.datasets import ConcatDataset

writer = SummaryWriter(comment="_evaluation_phase")

# Get thar args from the user
opt = arg_parser()

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2**4)
patch = (1, patch, patch)

cuda = True if torch.cuda.is_available() else False

# Loss function
task_loss = torch.nn.CrossEntropyLoss()
l1_loss = torch.nn.L1Loss()

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

resnet_background_classifier = models.resnet18(pretrained=True)
num_ftrs = resnet_background_classifier.fc.in_features
for param in resnet_background_classifier.parameters():
    param.requires_grad = False
resnet_background_classifier.fc = nn.Linear(num_ftrs, opt.n_classes)

resnet_classifier = resnet_classifier.cuda()
resnet_source_classifier = resnet_source_classifier.cuda()
resnet_target_classifier = resnet_target_classifier.cuda()
resnet_target_source_classifier = resnet_target_source_classifier.cuda()
resnet_target_fake_classifier = resnet_target_fake_classifier.cuda()
resnet_background_classifier = resnet_background_classifier.cuda()
seg_map_generator = SegmentationMapGenerator(opt)
########################################################################################################################

if cuda:
    generator.cuda()
    classifier.cuda()
    task_loss.cuda()
    l1_loss.cuda()
    seg_map_generator.cuda()

# Initialize weights
generator.load_state_dict(torch.load('models_ckpt/evalutation_phase.pt'))
# generator.train()
classifier.apply(weights_init_normal)
seg_map_generator.apply(weights_init_normal)

data_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

syn_image_folder = datasets.ImageFolder(root='dataset/linemod/synthetic/rgb', transform=data_transform)
depth_image_folder = datasets.ImageFolder(root='dataset/linemod/synthetic/depth',
                                          transform=transforms.Compose([
                                             transforms.Resize((opt.img_size, opt.img_size)),
                                             transforms.ToTensor()
                                          ]))
ori_dataset = datasets.ImageFolder(root='dataset/linemod/real/rgb', transform=data_transform)

background_dataset = datasets.ImageFolder(root='dataset/backgrounds', transform=data_transform)

syn_dataset = ConcatDataset(syn_image_folder, depth_image_folder)

test_dataset = datasets.ImageFolder(root='dataset/linemod/test/rgb', transform=data_transform)

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

background_loader = torch.utils.data.DataLoader(background_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                drop_last=True)

# Optimizers
optimizer_C = torch.optim.Adam(classifier.parameters(), lr=0.00002, betas=(opt.b1, opt.b2))
optimizer_fake = torch.optim.Adam(resnet_classifier.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_source = torch.optim.Adam(resnet_source_classifier.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_target = torch.optim.Adam(resnet_target_classifier.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_target_fake = torch.optim.Adam(resnet_target_fake_classifier.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_target_source = torch.optim.Adam(resnet_target_source_classifier.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_SF = torch.optim.Adam(seg_map_generator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_backgrounds = torch.optim.Adam(resnet_target_source_classifier.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))

confusion_matrix = torch.zeros(opt.n_classes, opt.n_classes)

for epoch in range(opt.n_epochs):
    for i, ((synth_images, synth_depth_images), (real_images, real_labels)) in enumerate(zip(syn_loader, ori_loader)):
        start_time = time.time()
        batches_done = len(syn_loader) * epoch + i

        synthetic_images, synthetic_labels = synth_images
        synthetic_depth_imgs, synthetic_depth_labels = synth_depth_images
        test_images, test_labels = next(iter(test_loader))
        background_images, background_labels = next(iter(background_loader))

        batch_size = synthetic_images.size(0)
        batch_size_m = real_images.size(0)
        valid = torch.ones(opt.batch_size, *patch)
        fake = torch.zeros(opt.batch_size, *patch)

        # Configure input
        synthetic_images = synthetic_images.cuda()
        synthetic_labels = synthetic_labels.cuda()
        real_images = real_images.cuda()
        synthetic_depth_imgs = synthetic_depth_imgs.cuda()
        background_images = background_images.cuda()
        real_labels = real_labels.cuda()
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()
        valid = valid.cuda()
        fake = fake.cuda()

        foreground_mask = synthetic_depth_imgs.clone()
        backgroud_mask = synthetic_depth_imgs.clone()
        m_foreground = foreground_mask <= 0.9
        m_background = foreground_mask > 0.9
        foreground_mask[m_foreground] = 1
        foreground_mask[m_background] = 0
        synthetic_images[m_background] = 0
        background_images[m_foreground] = 0
        background_images = background_images + synthetic_images

        onehot_syn = one_hot_embedding(synthetic_labels, opt.n_classes).cuda()
        onehot_real = one_hot_embedding(real_labels, opt.n_classes).cuda()

        # Generator and task training ############################################################
        optimizer_fake.zero_grad()
        z = torch.FloatTensor(opt.batch_size, opt.latent_dim).uniform_(-1, 1).cuda()
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
        target_label_pred = resnet_target_fake_classifier(fake_images.detach())
        target_only_loss = task_loss(target_label_pred, synthetic_labels)
        target_only_loss.backward()
        optimizer_target_fake.step()

        optimizer_target_fake.zero_grad()
        target_label_pred = resnet_target_fake_classifier(real_images)
        target_only_loss = task_loss(target_label_pred, real_labels)
        target_only_loss.backward()
        optimizer_target_fake.step()

        optimizer_target_source.zero_grad()
        target_label_pred = resnet_target_source_classifier(real_images)
        target_only_loss = task_loss(target_label_pred, real_labels)
        target_only_loss.backward()
        optimizer_target_source.step()

        optimizer_target_source.zero_grad()
        target_label_pred = resnet_target_source_classifier(synthetic_images)
        target_only_loss = task_loss(target_label_pred, synthetic_labels)
        target_only_loss.backward()
        optimizer_target_source.step()

        optimizer_backgrounds.zero_grad()
        target_label_pred = resnet_background_classifier(background_images)
        target_only_loss = task_loss(target_label_pred, synthetic_labels)
        target_only_loss.backward()
        optimizer_backgrounds.step()

        ##########################################################################################

        # Segmentation Generator using fake ######################################################
        optimizer_SF.zero_grad()
        fake_seg_map = seg_map_generator(fake_images.detach())
        seg_loss = l1_loss(fake_seg_map, synthetic_depth_imgs)
        seg_loss.backward()
        optimizer_SF.step()
        ##########################################################################################

        # Fresh classifier #######################################################################
        optimizer_C.zero_grad()
        fresh_label_pred = classifier(fake_images.detach())
        task_specific_loss = task_loss(fresh_label_pred, synthetic_labels)
        task_specific_loss.backward()
        optimizer_C.step()
        ##########################################################################################

        # Evaluation metrics #####################################################################
        fresh_pred = classifier(test_images)
        train_pred = resnet_classifier(test_images)
        source_pred = resnet_source_classifier(test_images)
        target_pred = resnet_target_classifier(test_images)
        target_fake_pred = resnet_target_fake_classifier(test_images)
        target_source_pred = resnet_target_source_classifier(test_images)
        background_pred = resnet_background_classifier(test_images)

        fresh_pred = np.argmax(fresh_pred.data.cpu().numpy(), axis=1)
        train_pred = np.argmax(train_pred.data.cpu().numpy(), axis=1)
        test_label = test_labels.data.cpu().numpy()
        source_pred = np.argmax(source_pred.data.cpu().numpy(), axis=1)
        target_pred = np.argmax(target_pred.data.cpu().numpy(), axis=1)
        target_fake_pred = np.argmax(target_fake_pred.data.cpu().numpy(), axis=1)
        target_source_pred = np.argmax(target_source_pred.data.cpu().numpy(), axis=1)
        background_pred = np.argmax(background_pred.data.cpu().numpy(), axis=1)

        acc_fresh = np.mean(fresh_pred == test_label)
        acc_synthetic = np.mean(train_pred == test_label)
        acc_source = np.mean(source_pred == test_label)
        acc_target = np.mean(target_pred == test_label)
        acc_target_fake = np.mean(target_fake_pred == test_label)
        acc_target_source = np.mean(target_source_pred == test_label)
        acc_background = np.mean(background_pred == test_label)
        confusion_matrix[train_pred, test_label] += 1

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
        writer.add_scalar('Accuracy Background', acc_background, batches_done)
        writer.add_scalar('Fresh classifier', acc_fresh, batches_done)

        batches_done = len(syn_loader) * epoch + i

        if batches_done % 100 == 0:
            test_seg_map = seg_map_generator(test_images)
            seg_back = test_seg_map >= 0.8
            seg_front = test_seg_map < 0.8
            test_seg_map[seg_back] = 0
            test_seg_map[seg_front] = 1
            sample = torch.cat((synthetic_images, fake_images, test_images, torch.mul(test_seg_map, test_images)))
            sample = make_grid(sample, normalize=True)
            writer.add_image("Images", sample, batches_done)
