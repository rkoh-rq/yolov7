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
from models.head import ResizeHead, ResizeHeadClass, ResizeHeadSigmoid
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

import pickle
def test(data,
         weights=None,
         batch_size=1,  # load one at a time
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
         v5_metric=False,
         resize=False,
         resize_class=False,
         resize_ood=False,
         switch_class=False,
         switch_ood=False):
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)
    task = "val"
    # Configure
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    dataloader = create_dataloader(data[task], imgsz, batch_size, 32, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]
    if (resize_class or resize_ood or resize):
        datasets = {}
        for i in [320, imgsz]: #range(192, imgsz+1, 32):
            datasets[i] = create_dataloader(data[task], i, batch_size, 32, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[1]

    if (resize):
        head = ResizeHead(1).to(device)
        head.half()
        checkpoint = torch.load("runs/train/head")
        head.load_state_dict(checkpoint)
        head.eval()
    if (resize_class or switch_class):
        head = ResizeHeadClass(4).to(device)
        head.half()
        checkpoint = torch.load("runs/train/head_class")
        head.load_state_dict(checkpoint)
        head.eval()
    if (resize_ood or switch_ood):
        head = ResizeHeadSigmoid().to(device)
        head.half()
        checkpoint = torch.load("runs/train/head_ood")
        head.load_state_dict(checkpoint)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if (switch_class):
        models = [model]
        models.append(attempt_load(weights[0].replace('cityscapes8','rainy'), map_location=device))
        models.append(attempt_load(weights[0].replace('cityscapes8','darker'), map_location=device))
        models.append(attempt_load(weights[0].replace('cityscapes8','brighter'), map_location=device))
    if (switch_ood):
        models = [model]
        models.append(attempt_load(weights[0].replace('cityscapes8', 'mixed4'), map_location=device))

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    if trace:
        if (switch_class or switch_ood):
            for i in range(len(models)):
                models[i] = TracedModel(models[i], device, imgsz)
        else:
            model = TracedModel(model, device, imgsz)

    
    with open('image_labels_val.pickle', 'rb') as handle:
        class_labels = pickle.load(handle)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        if (switch_class or switch_ood):
            for model in models:
                model.half()
        else:
            model.half()

    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    avg_size = 0

    label_to_int = {'none':0, 'rain':1, 'dark':2, 'bright':3}
    ood_stats = {'none': [], 'rain': [], 'dark': [], 'bright': []}

    for di, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        with torch.no_grad():
            if (switch_class):
                output = head(img)[0]
                output_l = torch.argmax(output)
                # output_l = label_to_int[class_labels[paths[0]]]
            elif (switch_ood):
                output = head(img)[0]
                output_l = 1 if output > 0.5 else 0
            if (resize_class): 
                output = head(img)[0]
                output_l = torch.argmax(output)
                resz = 320 if output_l == 0 else imgsz  # 1.
                img = datasets[resz][di][0].unsqueeze(0).to(device)
                img = img.half() if half else img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                avg_size += resz
            elif (resize_ood):
                output = head(img)[0]
                resz = 320 if output < 0.5 else imgsz  # 1.
                img = datasets[resz][di][0].unsqueeze(0).to(device)
                img = img.half() if half else img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                avg_size += resz
            elif (resize):
                output = head(img)[0]
                resz = int(torch.clamp(torch.round((640-192)/32 * output) * 32, 192, 640).item())
                img = datasets[resz][di][0].unsqueeze(0).to(device)
                img = img.half() if half else img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                avg_size += resz
            # Run model
            if (switch_class or switch_ood):
                model = models[output_l]  # inference and training outputs
            t = time_synchronized()
            out, train_out = model(img)  # inference and training outputs
            t0 += time_synchronized() - t
            nb, _, height, width = img.shape  # batch size, channels, height, width

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                ood_label = class_labels[str(path)]

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        ood_stats[ood_label].append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Append to text file
                if save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Append to pycocotools JSON dictionary
                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                    'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if plots:
                        confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                ood_stats[ood_label].append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    if resize or resize_class or resize_ood:
        print(avg_size/len(dataloader))

    for i in ood_stats:
        # Compute statistics
        ood_stat_i = ood_stats[i]
        ood_stat_i = [np.concatenate(x, 0) for x in zip(*ood_stat_i)]  # to numpy
        if len(ood_stat_i) and ood_stat_i[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*ood_stat_i, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(ood_stat_i[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % (i, seen, nt.sum(), mp, mr, map50, map))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        if "cityscapes" in data:
            anno_json = "./cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
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
    parser.add_argument('--resize', action='store_true', help='do resizing with regressor only. did not use in the end')
    parser.add_argument('--resize-class', action='store_true', help='do resizing based on class')
    parser.add_argument('--resize-ood', action='store_true', help='do resizing based on ood')
    parser.add_argument('--switch-class', action='store_true', help='switch model weights')
    parser.add_argument('--switch-ood', action='store_true', help='switch model weights if ood')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
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
             v5_metric=opt.v5_metric,
             resize=opt.resize,
             resize_class=opt.resize_class,
             resize_ood=opt.resize_ood,
             switch_class=opt.switch_class,
             switch_ood=opt.switch_ood,
             )