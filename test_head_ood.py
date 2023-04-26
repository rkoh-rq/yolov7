import argparse
import json
import os
from pathlib import Path
from threading import Thread

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

def test(data,
         batch_size=32,
         imgsz=640,
         dataloader=None):
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)
    with open('image_labels_val.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    task = "val"
    # Configure
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    dataloader = create_dataloader(data[task], imgsz, batch_size, 32, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]
    head = ResizeHeadSigmoid().to(device)
    checkpoint = torch.load("runs/train/head_ood")
    head.load_state_dict(checkpoint)
    head.eval()
    label_to_class = {'none': 0, 'rain': 1, 'dark': 1, 'bright': 1}

    a0p0, a0p1, a1p0, a1p1 = [0 for i in range(11)], [0 for i in range(11)], [0 for i in range(11)], [0 for i in range(11)]
    correct = [0 for i in range(11)]

    for img, _, paths, _ in tqdm(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = torch.Tensor([label_to_class[labels[p]] for p in paths]).to(device)
        output = head(img)
        print(output, targets)
        for i in range(1, 11):
            correct[i] += torch.sum((output.squeeze(1) < i/10) & (targets == 0)) + torch.sum((output.squeeze(1) >= i/10) & (targets == 1))
            a0p0[i] += torch.sum((output.squeeze(1) < i/10) & (targets == 0))
            a0p1[i] += torch.sum((output.squeeze(1) > i/10) & (targets == 0))
            a1p0[i] += torch.sum((output.squeeze(1) < i/10) & (targets == 1))
            a1p1[i] += torch.sum((output.squeeze(1) > i/10) & (targets == 1))
    print(a0p0, a0p1, a1p0, a1p1)
    print(correct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_head.py')
    parser.add_argument('--data', type=str, default='data/cityscapes_coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file

    test(opt.data,
          opt.batch_size,
          opt.img_size,
          )