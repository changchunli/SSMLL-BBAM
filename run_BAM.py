import argparse
import json
import os
import random
import sys
import time

import _init_paths

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data as Data
from lib.dataset.get_dataset import TransformUnlabeled_WS, get_datasets
from lib.dataset.handlers import (COCO2014_mask_handler, COCO2014_mm_handler,
                              NUS_WIDE_mask_handler, NUS_WIDE_mm_handler,
                              VOC2012_mask_handler, VOC2012_mm_handler, 
                              AWA_mask_handler, AWA_mm_handler)
from lib.models.MLDResnet import resnet50_ml_decoder
from lib.models.Resnet import create_model
from scipy.spatial.distance import cdist
from sklearn import metrics
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from lib.utils.helper import (ModelEma, add_weight_decay, clean_state_dict,
                          function_mAP, get_raw_dict)
from lib.utils.logger import setup_logger
from lib.utils.losses import (BBAM, AsymmetricLoss, BAMLoss, applyBBAM,
                          updateBbamPara)
from lib.utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter

from evaluation import classification_metrics_ml

np.set_printoptions(suppress=True, precision=4)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MASK_HANDLER_DICT = {
    'VOC2012': VOC2012_mask_handler,
    'coco': COCO2014_mask_handler,
    # 'nuswide': NUS_WIDE_mask_handler,
    'Animals_with_Attributes2': AWA_mask_handler
}

MM_HANDLER_DICT = {
    'VOC2012': VOC2012_mm_handler,
    'coco': COCO2014_mm_handler,
    # 'nuswide': NUS_WIDE_mm_handler,
    'Animals_with_Attributes2': AWA_mm_handler
}

NUM_CLASS = {'VOC2012': 20, 'coco': 80, 'Animals_with_Attributes2': 85}


def parser_args():
    parser = argparse.ArgumentParser(description='Main')

    # data
    parser.add_argument('--dataset_name',
                        default='VOC2012',
                        choices=['VOC2012', 'coco', 'Animals_with_Attributes2'],
                        help='dataset name')
    parser.add_argument('--dataset_dir',
                        default='Dataset',
                        metavar='DIR',
                        help='dir of all datasets')
    parser.add_argument('--img_size',
                        default=224,
                        type=int,
                        help='size of input images')
    parser.add_argument('--output',
                        default='SSMLL-BAM/outputs',
                        metavar='DIR',
                        help='path to output folder')

    # train
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs',
                        default=40,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b',
                        '--warmup_batch_size',
                        default=16,
                        type=int,
                        help='batch size for warmup')
    parser.add_argument('--lr',
                        '--learning_rate',
                        default=1e-4,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--wd',
                        '--weight_decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('-p',
                        '--print_freq',
                        default=400,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--amp',
                        action='store_true',
                        default=True,
                        help='apply amp')
    parser.add_argument('--early_stop',
                        action='store_true',
                        default=True,
                        help='apply early stop')
    parser.add_argument('--save_PR',
                        action='store_true',
                        default=False,
                        help='on/off PR')
    parser.add_argument('--optim',
                        default='adamw',
                        type=str,
                        help='optimizer used')
    parser.add_argument('--warmup_epochs',
                        default=12,
                        type=int,
                        help='the number of epochs for warmup')
    parser.add_argument('--lb_ratio',
                        default=0.05,
                        type=float,
                        help='the ratio of lb:(lb+ub)')

    #loss
    parser.add_argument('--loss',
                        default='BAM',
                        type=str,
                        help='used_loss for all')
    parser.add_argument('--cos-margin',
                        default=0.4,
                        type=float,
                        metavar='N',
                        help='margin for cosine loss')
    parser.add_argument('--cos-norm',
                        default=20.0,
                        type=float,
                        help='norm for cosine loss')

    parser.add_argument('--start_bbam_epoch',
                        default=12,
                        type=int,
                        help='the number of epochs for warmup')

    #negtive sampling
    parser.add_argument('--eta', default=5, type=float, help='eta') 
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha')

    parser.add_argument('--gamma', default=0.01, type=float, help='gamma')

    parser.add_argument('--cutout',
                        default=0.5,
                        type=float,
                        help='cutout factor')

    parser.add_argument('--init_pos_per',
                        default=1.0,
                        type=float,
                        help='init pos_per')
    parser.add_argument('--init_neg_per',
                        default=1.0,
                        type=float,
                        help='init neg_per')

    # random seed
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='seed for initializing training. ')

    # model
    parser.add_argument('--net',
                        default='resnet50',
                        type=str,
                        choices=['resnet50', 'mlder'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--is_data_parallel',
                        action='store_true',
                        default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument('--ema_decay',
                        default=0.9997,
                        type=float,
                        metavar='M',
                        help='decay of model ema')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    if args.lb_ratio == 0.05:
        args.lb_bs = 4
        args.ub_bs = 60 #60
    elif args.lb_ratio == 0.1:
        args.lb_bs = 8
        args.ub_bs = 56 # 56
    elif args.lb_ratio == 0.15:
        args.lb_bs = 16
        args.ub_bs = 48 # 52
    elif args.lb_ratio == 0.2:
        args.lb_bs = 32
        args.ub_bs = 32 #48

    if args.dataset_name == 'VOC2012':
        args.lb_bs = int(args.lb_bs / 2)
        args.ub_bs = int(args.ub_bs / 2)
    
    if args.dataset_name == 'Animals_with_Attributes2':
        args.lb_bs = 32
        args.ub_bs = 96

    args.output = args.net + '_outputs'
    args.resume = '%s_outputs/%s/%s/%s/warmup_%s_%s_%s_%s_%s_%s/warmup_model.pth.tar' % (
        args.net, args.dataset_name, args.img_size, args.lb_ratio, args.loss,
        args.warmup_epochs, args.cos_margin, args.cos_norm, args.eta, args.alpha)
    args.n_classes = NUM_CLASS[args.dataset_name]
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    args.output = os.path.join(
        args.output, args.dataset_name, '%s' % args.img_size,
        '%s' % args.lb_ratio,
        'BBAM_%s_%s_%s_%s_%s_%s_%s_%s' % (args.warmup_epochs, args.epochs,
                                 args.cos_margin, args.cos_norm, args.eta, args.gamma, args.lr, args.alpha))

    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: " + ' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)


def main_worker(args, logger):
    # build model
    if args.net in ['resnet50']:
        model = create_model(args.net, n_classes=args.n_classes)
    elif args.net == 'mlder':
        model = resnet50_ml_decoder(num_classes=args.n_classes)

    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()

    if args.resume:
        # print(args.resume)
        if os.path.exists(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume))

            args.start_epoch = checkpoint['epoch'] + 1
            if 'state_dict' in checkpoint and 'state_dict_ema' in checkpoint:
                if args.dataset_name in ['Animals_with_Attributes2']:
                    state_dict = clean_state_dict(checkpoint['state_dict_ema'])
                else:
                    state_dict = clean_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("No model or state_dicr Found!!!")

            model.load_state_dict(state_dict, strict=False)
            print(np.array(checkpoint['regular_mAP']))
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))

            del checkpoint
            del state_dict
            # del state_dict_ema
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ema_m = ModelEma(model, args.ema_decay)

    # Data loading code
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args)
    print("len(lb_train_dataset):", len(lb_train_dataset))
    print("len(ub_train_dataset):", len(ub_train_dataset))
    print("len(val_dataset):", len(val_dataset))

    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset,
        batch_size=args.warmup_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=256,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    macroF1s = AverageMeter('macro_F1', ':5.5f', val_only=True)
    microF1s = AverageMeter('micro_F1', ':5.5f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    macroF1s_ema = AverageMeter('macroF1_ema', ':5.5f', val_only=True)
    microF1s_ema = AverageMeter('microF1_ema', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs, [eta, epoch_time, macroF1s, microF1s, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # optimizer
    optimizer = set_optimizer(model, args)
    args.steps_per_epoch = len(lb_train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=args.lr,
                                        steps_per_epoch=args.steps_per_epoch,
                                        epochs=args.warmup_epochs,
                                        pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_macroF1 = 0
    best_regular_microF1 = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    best_ema_macroF1 = 0
    best_ema_microF1 = 0
    regular_mAP_list = []
    regular_macroF1_list = []
    regular_microF1_list = []
    ema_mAP_list = []
    ema_macroF1_list = []
    ema_microF1_list = []
    best_mAP = 0

    results_bbam_all = []
    results_all = []
    results_bbam_ema_all = []
    results_ema_all = []

    criterion_lb = BAMLoss()
    criterion_ub = BAMLoss()

    # centers = torch.zeros(args.n_classes, 256).cuda()
    centers = torch.zeros(args.n_classes, model.in_features).cuda()
    means = torch.zeros(2, args.n_classes).cuda()
    variances = torch.zeros(2, args.n_classes).cuda()
    mean_variances = torch.zeros(args.n_classes).cuda()

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):

        torch.cuda.empty_cache()

        # train for one epoch
        if epoch < args.warmup_epochs:
            loss = train(lb_train_loader, model, ema_m, optimizer, scheduler,
                         epoch, args, logger, criterion_lb, means, variances,
                         mean_variances)
        else:
            adjust_per(epoch - args.warmup_epochs, args)
            print('Pos per: %.3f Neg per: %.3f' % (args.pos_per, args.neg_per))
            pb_train_dataset = pseudo_label(ub_train_dataset, model,
                                            ema_m.module, epoch, args, logger,
                                            means, variances, mean_variances)
            (centers, means, variances, mean_variances, features,
             targets) = updatebbampara(lb_train_dataset, pb_train_dataset,
                                       model, ema_m.module, epoch, args,
                                       logger, centers, means, variances,
                                       mean_variances)
            lb_train_dataset_ns, pb_train_dataset_ns = negative_sampling(
                lb_train_dataset, pb_train_dataset, centers, features, targets,
                args)
            pb_train_loader = torch.utils.data.DataLoader(
                pb_train_dataset_ns,
                batch_size=args.ub_bs,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)
            if epoch == args.warmup_epochs:
                lb_train_loader = torch.utils.data.DataLoader(
                    lb_train_dataset_ns,
                    batch_size=args.lb_bs,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True)
                optimizer = set_optimizer(model, args)
                args.steps_per_epoch = len(pb_train_loader)
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.lr,
                    steps_per_epoch=args.steps_per_epoch,
                    epochs=args.epochs,
                    pct_start=0.2)

            loss = semi_train(lb_train_loader, pb_train_loader, model, ema_m,
                              optimizer, scheduler, epoch, args, logger,
                              criterion_lb, criterion_ub, means, variances,
                              mean_variances)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('pos_per', args.pos_per, epoch)
                summary_writer.add_scalar('neg_per', args.neg_per, epoch)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate',
                                      optimizer.param_groups[0]['lr'], epoch)

        # evaluate on validation set
        results_bbam, results = validate(val_loader, model, epoch,
                                        args, logger, means, variances,
                                        mean_variances)
        results_bbam_ema, results_ema = validate(val_loader, ema_m.module,
                                                epoch, args, logger,
                                                means, variances,
                                                mean_variances)

        results_bbam_all.append(results_bbam)
        results_all.append(results)
        results_bbam_ema_all.append(results_bbam_ema)
        results_ema_all.append(results_ema)

        macroF1, microF1, mAP = results_bbam[5], results_bbam[6], results_bbam[3]  #11
        macroF1_ema, microF1_ema, mAP_ema = results_bbam_ema[5], results_bbam_ema[6], results_bbam_ema[3]

        macroF1s.update(macroF1)
        microF1s.update(microF1)
        mAPs.update(mAP)
        macroF1s_ema.update(macroF1_ema)
        microF1s_ema.update(microF1_ema)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

        regular_macroF1_list.append(macroF1)
        regular_microF1_list.append(microF1)
        regular_mAP_list.append(mAP)
        ema_macroF1_list.append(macroF1_ema)
        ema_microF1_list.append(microF1_ema)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_macroF1', macroF1, epoch)
            summary_writer.add_scalar('val_macroF1_ema', macroF1_ema, epoch)
            summary_writer.add_scalar('val_microF1', microF1, epoch)
            summary_writer.add_scalar('val_microF1_ema', microF1_ema, epoch)
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = max(best_regular_mAP, mAP)
            best_regular_epoch = epoch
            best_regular_macroF1 = macroF1
            best_regular_microF1 = microF1
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = max(mAP_ema, best_ema_mAP)
            best_ema_epoch = epoch
            best_ema_macroF1 = macroF1_ema
            best_ema_microF1 = microF1_ema

        if mAP_ema > mAP:
            mAP = mAP_ema

        is_best = mAP > best_mAP
        if is_best:
            best_epoch = epoch
            best_mAP = mAP
            state_dict = model.state_dict()
            state_dict_ema = ema_m.module.state_dict()
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'state_dict_ema': state_dict_ema,
                    'regular_mAP': regular_mAP_list,
                    'ema_mAP': ema_mAP_list,
                    'regular_macroF1': regular_macroF1_list,
                    'ema_macroF1': ema_macroF1_list,
                    'regular_microF1': regular_microF1_list,
                    'ema_microF1': ema_microF1_list,
                    'best_regular_mAP': best_regular_mAP,
                    'best_ema_mAP': best_ema_mAP,
                    'best_regular_macroF1': best_regular_macroF1,
                    'best_ema_macroF1': best_ema_macroF1,
                    'best_regular_microF1': best_regular_microF1,
                    'best_ema_microF1': best_ema_microF1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=True,
                filename=os.path.join(args.output, 'best_model.pth.tar'))

        logger.info("{} | Set best mAP {} in ep {}".format(
            epoch, best_mAP, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(
            best_regular_mAP, best_regular_epoch))

        # early stop

        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch,
                                               best_regular_epoch) > 5:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info(
                        "epoch - best_epoch = {}, stop!".format(epoch -
                                                                best_epoch))
                    break

    print("Best mAP:", best_mAP)

    np.save(args.output + "/train_results.npy",
            (np.array(results_bbam_all), np.array(results_all)))
    np.save(args.output + "/train_results_ema.npy",
            (np.array(results_bbam_ema_all), np.array(results_ema_all)))

    if summary_writer:
        summary_writer.close()

    return 0


def adjust_per(epoch, args):

    if epoch == 0:
        args.pos_per = args.init_pos_per
        args.neg_per = args.init_neg_per

    args.pos_per = np.clip(args.pos_per, 0.0, 1.0)
    args.neg_per = np.clip(args.neg_per, 0.0, 0.998)


def set_optimizer(model, args):

    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            params=parameters, lr=args.lr,
            weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [
            {
                "params":
                [p for n, p in model.named_parameters() if p.requires_grad]
            },
        ]
        optimizer = getattr(torch.optim,
                            'AdamW')(param_dicts,
                                     args.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=args.weight_decay)


    return optimizer


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


@torch.no_grad()
def updatebbampara(lb_train_dataset, ub_train_dataset, model, ema_model, epoch,
                   args, logger, centers, means, variances, mean_variances):

    model.eval()

    features_labeled = torch.Tensor([]).cuda()
    targets_labeled = torch.Tensor([]).cuda()
    features_unlabeled = torch.Tensor([]).cuda()
    targets_unlabeled = torch.Tensor([]).cuda()

    unlabeled_loader = Data.DataLoader(dataset=ub_train_dataset,
                                       batch_size=256,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       pin_memory=True)
    for i, ((inputs_w, inputs_s), targets,
            mask) in enumerate(unlabeled_loader):

        inputs_w, targets = inputs_w.cuda(non_blocking=True), targets.cuda()

        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            (logits_w, feature) = model(inputs_w, flag=1)

        targets[mask == 0] = -1
        features_unlabeled = torch.cat([features_unlabeled, feature], dim=0)
        targets_unlabeled = torch.cat([targets_unlabeled, targets], dim=0)

    labeled_loader = Data.DataLoader(dataset=lb_train_dataset,
                                     batch_size=256,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     pin_memory=True)
    # Updating parameters of balancing categorical variances
    for i, ((inputs_w, inputs_s), targets) in enumerate(labeled_loader):
        inputs_w, targets = inputs_w.cuda(non_blocking=True), targets.cuda()

        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            (_, feature) = model(inputs_w, flag=1)

        features_labeled = torch.cat([features_labeled, feature], dim=0)
        targets_labeled = torch.cat([targets_labeled, targets], dim=0)

    features = torch.cat([features_labeled, features_unlabeled], dim=0)
    targets = torch.cat([targets_labeled, targets_unlabeled], dim=0)
    centers, means, variances, mean_variances = updateBbamPara(
        features, targets, centers, means, variances, mean_variances, epoch,
        args.gamma)

    print("Means: ", means.data)
    print("Variances: ", variances.data)
    print("Mean of variances: ", mean_variances.data)

    return centers, means, variances, mean_variances, features, targets


@torch.no_grad()
def negative_sampling(lb_train_dataset, ub_train_dataset, centers, features,
                      targets, args):
    targets = targets.cpu().numpy()

    cos_sim = 1. - cdist(features.cpu().numpy(), centers.cpu().numpy(), 'cosine')
    
    mask_lb = np.ones_like(lb_train_dataset.Y)
    mask_ub = ub_train_dataset.Mask
    mask = np.vstack((mask_lb, mask_ub))
 
    cos_sim[targets == 1] = -1
    cos_sim[mask == 0] = -1
 
    num_pos_label = np.sum(targets, axis=0)
    num_neg_label = np.sum((1 - targets) * mask, axis=0)
    num_neg_sam = np.minimum(args.eta * num_pos_label, num_neg_label)
 
    sorted_cos_sim = -np.sort(-cos_sim, axis=0)
 
    indices = [int(x) - 1 for x in num_neg_sam]
    thre_vec = sorted_cos_sim[indices, range(cos_sim.shape[1])]

    masks_ns = np.zeros_like(targets)
    masks_ns[cos_sim >= thre_vec] = 1
    masks_ns[targets == 1] = 1
    # masks_ns = np.ones_like(targets)

    lb_train_dataset_ns = MASK_HANDLER_DICT[args.dataset_name](
        lb_train_dataset.X,
        lb_train_dataset.Y,
        masks_ns[:len(lb_train_dataset)],
        args.dataset_dir,
        transform=TransformUnlabeled_WS(args))

    pb_train_dataset_ns = MM_HANDLER_DICT[args.dataset_name](
        ub_train_dataset.X,
        ub_train_dataset.Y,
        ub_train_dataset.Mask,
        masks_ns[len(lb_train_dataset):],
        args.dataset_dir,
        transform=TransformUnlabeled_WS(args))

    return lb_train_dataset_ns, pb_train_dataset_ns


@torch.no_grad()
def pseudo_label(ub_train_dataset, model, ema_model, epoch, args, logger,
                 means, variances, mean_variances):

    loader = torch.utils.data.DataLoader(ub_train_dataset,
                                         batch_size=256,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=False)

    model.eval()
    outputs = []
    labels = []
    for i, ((inputs_w, inputs_s), targets) in enumerate(loader):

        inputs_w, targets = inputs_w.cuda(non_blocking=True), targets.cuda()

        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits_w = ema_model(inputs_w)

        logits_w_bbam = BBAM(logits_w, targets, means, variances, mean_variances,
                             epoch - 1, args)
        outputs.append(torch.sigmoid(logits_w_bbam).detach().cpu().numpy())
        labels.append(targets.cpu().numpy())

    labels = np.concatenate(labels)
    outputs = np.concatenate(outputs)
    sorted_outputs = -np.sort(-outputs, axis=0)

    n_ub = len(outputs)

    indices = [int(x) - 1 for x in args.pos_label_freq * n_ub]
    thre_vec = sorted_outputs[indices, range(outputs.shape[1])]

    pseudo_labels = (outputs >= thre_vec).astype(np.float32)

    mask = np.zeros_like(pseudo_labels)

    pos_indices = [
        int(x) - 1 for x in args.pos_per * args.pos_label_freq * n_ub
    ]
    pos_thre = sorted_outputs[pos_indices, range(outputs.shape[1])]

    mask[outputs >= pos_thre] = 1

    neg_indices = [
        int(x) - 1 for x in args.neg_per * args.neg_label_freq * n_ub
    ]
    sorted_outputs_neg = np.sort(outputs, axis=0)
    neg_thre = sorted_outputs_neg[neg_indices, range(outputs.shape[1])]

    mask[outputs <= neg_thre] = 1

    ub_train_imgs = ub_train_dataset.X
    pb_train_dataset = MASK_HANDLER_DICT[args.dataset_name](
        ub_train_imgs,
        pseudo_labels,
        mask,
        args.dataset_dir,
        transform=TransformUnlabeled_WS(args))

    #########  Save (Train: OPs CPs ORs CRs)  #######################
    if args.save_PR:
        n_correct_pos = (labels * pseudo_labels).sum(0)
        n_pred_pos = ((pseudo_labels == 1)).sum(0)
        n_true_pos = labels.sum(0)
        OP = n_correct_pos.sum() / n_pred_pos.sum()
        CP = np.nanmean(n_correct_pos / n_pred_pos)
        OR = n_correct_pos.sum() / n_true_pos.sum()
        CR = np.nanmean(n_correct_pos / n_true_pos)

        auc = np.zeros(labels.shape[1])
        for i in range(labels.shape[1]):
            auc[i] = metrics.roc_auc_score(labels[:, i], pseudo_labels[:, i])
        AUC = np.nanmean(auc)

        logger.info('Train: ')
        logger.info(' AUC: %.3f' % AUC)
        logger.info(' OP: %.3f' % OP)
        logger.info(' CP: %.3f' % CP)
        logger.info(' OR: %.3f' % OR)
        logger.info(' CR: %.3f' % CR)

        PR.append([OP, CP, OR, CR, AUC])
        np.save(os.path.join(args.output, "PR"), np.array(PR))
    #############################################################

    return pb_train_dataset


PR = []


def semi_train(lb_train_loader, ub_train_loader, model, ema_m, optimizer,
               scheduler, epoch, args, logger, criterion_lb, criterion_ub,
               means, variances, mean_variances):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    loss_lb = AverageMeter('L_lb', ':5.3f')
    loss_ub = AverageMeter('L_ub', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(args.steps_per_epoch,
                             [loss_lb, loss_ub, lr, losses, mem],
                             prefix="Epoch: [{}/{}]".format(
                                 epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()
    lb_train_iter = iter(lb_train_loader)
    for i, ((inputs_w_ub, inputs_s_ub), labels_ub, mask,
            mask_ns_ub) in enumerate(ub_train_loader):

        try:
            (_, inputs_s_lb), labels_lb, mask_ns_lb = next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (_, inputs_s_lb), labels_lb, mask_ns_lb = next(lb_train_iter)

        n_lb = labels_lb.shape[0]
        n_ub = labels_ub.shape[0]
        inputs = torch.cat([inputs_s_lb, inputs_s_ub],
                           dim=0).cuda(non_blocking=True)
        labels = torch.cat([labels_lb, labels_ub],
                           dim=0).cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        mask_ns_lb = mask_ns_lb.cuda(non_blocking=True)
        mask_ns_ub = mask_ns_ub.cuda(non_blocking=True)

        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)

        logits_bbam, logits_bbam1 = applyBBAM(logits, labels, means, variances,
                                              mean_variances, epoch, args)

        logits_bbam_s_lb, logits_bbam_s_ub = logits_bbam[:n_lb], logits_bbam[
            n_lb:]
        labels_lb, labels_ub = labels[:n_lb], labels[n_lb:]

        L_lb = (criterion_lb(logits_bbam_s_lb, labels_lb) * mask_ns_lb).sum()
        L_ub = (criterion_ub(logits_bbam_s_ub, labels_ub) * mask *
                mask_ns_ub).sum()

        # L_lb = (criterion_lb(logits_bbam_s_lb, labels_lb)).sum()
        # L_ub = (criterion_ub(logits_bbam_s_ub, labels_ub) * mask).sum()

        loss = L_lb + L_ub

        # *********************************

        # record loss
        loss_lb.update(L_lb.item(), inputs_s_lb.size(0))
        loss_ub.update(L_ub.item(), inputs_s_ub.size(0))
        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args,
          logger, criterion, means, variances, mean_variances):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    loss_base = AverageMeter('L_base', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(args.steps_per_epoch,
                             [loss_base, lr, losses, mem],
                             prefix="Epoch: [{}/{}]".format(
                                 epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, ((inputs_w, inputs_s), targets) in enumerate(train_loader):

        # ****************compute loss************************

        batch_size = inputs_w.size(0)

        inputs = torch.cat([inputs_w, inputs_s], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)

        logits_w, logits_s = torch.split(logits[:], batch_size)

        logits_s_bbam, logits_s_bbam1 = applyBBAM(logits_s, targets, means,
                                                  variances, mean_variances,
                                                  epoch, args)

        L_base = criterion(logits_s_bbam, targets).sum()

        loss = L_base

        # record loss
        loss_base.update(L_base.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, epoch, args, logger, means, variances,
             mean_variances):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(len(val_loader), [batch_time, mem],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    outputs_total = np.array([])
    outputs_bbam_total = np.array([])
    targets_total = np.array([])

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(inputs)
            outputs_bbam = BBAM(outputs, 0.5 * torch.ones_like(targets), means, variances, mean_variances,
                                epoch, args)
            outputs = torch.sigmoid(outputs)
            outputs_bbam = torch.sigmoid(outputs_bbam)

        outputs_total = np.array(
            outputs.detach().cpu()) if batch_idx == 0 else np.vstack(
                (outputs_total, np.array(outputs.detach().cpu())))
        outputs_bbam_total = np.array(
            outputs_bbam.detach().cpu()) if batch_idx == 0 else np.vstack(
                (outputs_bbam_total, np.array(outputs_bbam.detach().cpu())))
        targets_total = np.array(
            targets.cpu()) if batch_idx == 0 else np.vstack(
                (targets_total, np.array(targets.detach().cpu())))

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx, logger)

    results = classification_metrics_ml(outputs_total, targets_total)

    print("Calculating performance:")
    logger.info(
        "Coverage: {}\tRanking Loss: {}\tRanking AP: {}\tpr_auc: {}\troc_auc: {}\tMacro F1: {}\tMicro F1: {}\t"
        "Hamming Loss: {}\tmacro_auc: {}\tmicro_auc: {}\tOne Loss: {}\tmAP: {}\n".format(
            results[0], results[1], results[2], results[3], results[4], results[5], results[6], 
             results[7], results[8], results[9], results[10], results[11]))

    results_bbam = classification_metrics_ml(outputs_bbam_total, targets_total)

    print("Calculating performance after bbam:")
    logger.info(
        "Coverage: {}\tRanking Loss: {}\tRanking AP: {}\tpr_auc: {}\troc_auc: {}\tMacro F1: {}\tMicro F1: {}\t"
        "Hamming Loss: {}\tmacro_auc: {}\tmicro_auc: {}\tOne Loss: {}\tmAP: {}\n".format(
            results_bbam[0], results_bbam[1], results_bbam[2], results_bbam[3], results_bbam[4], results_bbam[5], results_bbam[6], 
             results_bbam[7], results_bbam[8], results_bbam[9], results_bbam[10], results_bbam[11]))

    return results_bbam, results


if __name__ == '__main__':
    main()
