import torch
from torch import nn
from torch import optim
from torchvision import transforms
import numpy as np
import sys
import argparse
from models.vae import VAE, vae_loss

import argparse
import json
import os
from pathlib import Path
from threading import Thread
from torchvision.utils import save_image

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from models.head import ResizeHeadSigmoid
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

import pickle

import math
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def merge_images(images, size):
    # merge a mini-batch of images into a single grid of images
    H, W, C = images.shape[1], images.shape[2], images.shape[3]
    merged_img = np.zeros((H * size[0], W * size[1], C))

    for idx, img in enumerate(images):
        i = idx // size[1]  # row number
        j = idx % size[1]   # column number

        merged_img[H * i: H * (i+1), W * j: W * (j+1), :] = img

    return merged_img

def imsave(X, path):
    # save the batch of images in X as a single image in path
    grid_sizes = {4: (2, 2),
                  8: (2, 4),
                  16: (4, 4),
                  32: (4, 8),
                  64: (8, 8),
                  128: (8, 16),
                  256: (16, 16),
                  512: (16, 32),
                  1024: (32, 32)}
    N = X.shape[0]
    size = grid_sizes.get(N, (1, N))

    imgs = (X.to('cpu').numpy().transpose(0, 2, 3, 1) + 1.)/2.
    img = merge_images(imgs, size)
    plt.figure()
    plt.imshow(img)
    plt.savefig(path)
    plt.close()


parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--data-path', type=str,
                    default='/data/DB/celebA/img_align_celeba/',
                    help='path for the images dir')
parser.add_argument('--img-crop', type=int, default=148,
                    help='size for center cropping (default: 148)')
parser.add_argument('--img-resize', type=int, default=64,
                    help='size for resizing (default: 64)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--valid-split', type=float, default=.2,
                    help='fraction of data for validation (default: 0.2)')
parser.add_argument('--kl-weight', type=float, default=1e-3,
                    help='weight of the KL loss (default: 1e-3)')
parser.add_argument('--filters', type=str, default='64, 128, 256, 512',
                    help=('number of filters for each conv. layer (default: '
                          + '\'64, 128, 256, 512\')'))
parser.add_argument('--kernel-sizes', type=str, default='3, 3, 3, 3',
                    help=('kernel sizes for each conv. layer (default: '
                          + '\'3, 3, 3, 3\')'))
parser.add_argument('--strides', type=str, default='2, 2, 2, 2',
                    help=('strides for each conv. layer (default: \'2, 2, 2, '
                          + '2\')'))
parser.add_argument('--latent-dim', type=int, default=128,
                    help='latent space dimension (default: 128)')
parser.add_argument('--batch-norm', type=int, default=0,
                    help=('whether to use or not batch normalization (default:'
                          + ' 1)'))
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
args = parser.parse_args()
args.filters = [int(item) for item in args.filters.split(',')]
args.kernel_sizes = [int(item) for item in args.kernel_sizes.split(',')]
args.strides = [int(item) for item in args.strides.split(',')]
args.batch_norm = bool(args.batch_norm)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def train(vae, optimizer, train_loader, n_epochs, kl_weight=1e-3,
          valid_loader=None, n_gen=0):

    device = next(vae.parameters()).device
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))

        # training phase
        vae.train()  # training mode
        for i, (X, _, paths, _) in enumerate(train_loader):
            X = X[:, :, :, :64]
            X = X.to(device)
            X = X.float()
            X /= 255.0

            # forward pass
            Xrec, z_mean, z_logvar = vae(X)

            # loss, backward pass and optimization step
            loss, reconst_loss, kl_loss = vae_loss(Xrec, X, z_mean, z_logvar,
                                                   kl_weight=kl_weight)
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            sys.stdout.write(
              '\r'
              + '........{} mini-batch loss: {:.3f} |'
                .format(i + 1, loss.item())
              + ' reconst loss: {:.3f} |'
                .format(reconst_loss.item())
              + ' kl loss: {:.3f}'
                .format(kl_loss.item()))
            sys.stdout.flush()

        torch.save(vae.state_dict(), './models/vae.pth')

        # evaluation phase
        print()
        with torch.no_grad():
            vae.eval()  # inference mode

            # compute training loss
            train_loss = 0.
            for i, (X, _, paths, _) in enumerate(train_loader):
                X = X[:, :, :, :64]
                X = X.to(device)
                X = X.float()
                X /= 255.0

                Xrec, z_mean, z_logvar = vae(X)
                train_loss += vae_loss(Xrec, X, z_mean, z_logvar,
                                       kl_weight=kl_weight)[0]

                # save original and reconstructed images
                if i == 0:
                    imsave(X, './imgs/train_orig.png')
                    imsave(Xrec, './imgs/train_rec.png')

            train_loss /= i + 1
            print('....train loss = {:.3f}'.format(train_loss.item()))

            if valid_loader is None:
                print()
            else:  # compute validation loss
                valid_loss = 0.
                for i, (X, _, paths, _) in enumerate(valid_loader):
                    X = X[:, :, :, :64]
                    X = X.to(device)
                    X = X.float()
                    X /= 255.0

                    Xrec, z_mean, z_logvar = vae(X)
                    valid_loss += vae_loss(Xrec, X, z_mean, z_logvar,
                                           kl_weight=kl_weight)[0]

                    # save original and reconstructed images
                    if i == 0:
                        imsave(X, './imgs/valid_orig.png')
                        imsave(Xrec, './imgs/valid_rec.png')

                valid_loss /= i + 1
                print('....valid loss = {:.3f}'.format(valid_loss.item()))
                print()

            # generate some new examples
            if n_gen > 0:
                z = torch.randn((n_gen, vae.latent_dim)).to(device)
                Xnew = vae.decoder(z)
                imsave(Xnew, './imgs/gen.png')

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
parser.add_argument('--task', default='val', help='train, val, test, speed or study')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--verbose', action='store_true', help='report mAP by class')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument('--project', default='runs/test', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
parser.add_argument('--create-pickle', action='store_true', help='recreate filename.pickle')
opt = parser.parse_args()
opt.save_json |= opt.data.endswith('coco.yaml')


# Configure
if isinstance(opt.data, str):
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

check_dataset(data)  # check
with open('image_labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)
train_loader = create_dataloader(data['train'], 128, opt.batch_size, 1, opt, rect=True,
                                prefix=colorstr(f'train: '))[0]
valid_loader = create_dataloader(data['val'], 128, opt.batch_size, 1, opt, rect=True,
                                prefix=colorstr(f'val: '))[0]

img_channels = 3

vae = VAE(img_channels,
          args.img_resize,
          args.latent_dim,
          args.filters,
          args.kernel_sizes,
          args.strides,
          activation=nn.LeakyReLU,
          out_activation=nn.Tanh,
          batch_norm=args.batch_norm).to(DEVICE)
print(vae)

optimizer = optim.Adam(vae.parameters(),
                       lr=args.lr,
                       weight_decay=0.)

train(vae, optimizer, train_loader, args.epochs, kl_weight=args.kl_weight,
      valid_loader=valid_loader, n_gen=args.batch_size)