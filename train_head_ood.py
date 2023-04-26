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

def train(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False):
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    if compute_loss is None:
        compute_loss = ComputeLoss(model)  # init loss class
    if trace:
        model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

    check_dataset(data)  # check

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = 'train'  # path to train/val/test images
    with open('image_labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    dataloader = create_dataloader(data[task], 640, batch_size, gs, opt, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]
    label_to_class = {'none': 0, 'rain': 1, 'dark': 1, 'bright': 1}
    bce_loss = torch.nn.BCELoss()
    head = ResizeHeadSigmoid().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    running_loss = 0.0


    for epoch in range(20):
        for img, _, paths, _ in tqdm(dataloader):
            img = img.to(device, non_blocking=True)
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = torch.tensor([label_to_class[labels[p]] for p in paths], dtype=torch.int64).to(device)
            optimizer.zero_grad()
            output = head(img)
            loss = bce_loss(output, targets.type(torch.cuda.FloatTensor).unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'[{epoch + 1}] loss: {running_loss/len(dataloader):.10f}')
        running_loss = 0.0
    torch.save(head.state_dict(), "runs/train/head_ood_alt")


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_head.py')
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
    opt.data = check_file(opt.data)  # check file

    train(opt.data,
          opt.weights,
          opt.batch_size,
          opt.img_size,
          opt.conf_thres,
          opt.iou_thres,
          opt.save_json,
          opt.single_cls,
          opt.augment,
          opt.verbose,
          save_txt=opt.save_txt | opt.save_hybrid,
          save_hybrid=opt.save_hybrid,
          save_conf=opt.save_conf,
          trace=not opt.no_trace,
          v5_metric=opt.v5_metric
          )