import argparse
import json
import os
import random
import sys
import time

import _init_paths

import numpy as np
import torch
import torch.optim
import torch.utils.data as Data
from lib.dataset.get_dataset import get_datasets
from lib.models.MLDResnet import resnet50_ml_decoder
from lib.models.Resnet import create_model
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from lib.utils.helper import (ModelEma, add_weight_decay, clean_state_dict,
                          function_mAP, get_raw_dict)
from lib.utils.logger import setup_logger
from lib.utils.losses import AsymmetricLoss, BAMLoss, applyBBAM, BBAM, updateBbamPara
from lib.utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter

from evaluation import classification_metrics_ml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=4)

NUM_CLASS = {'VOC2012': 20, 'coco': 80, 'Animals_with_Attributes2': 85}


def parser_args():
    parser = argparse.ArgumentParser(description='Warmup Stage')

    # data
    parser.add_argument('--dataset_name',
                        default='nuswide',
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
                        default=20,
                        type=float,
                        help='norm for cosine loss')

    parser.add_argument('--start_bbam_epoch',
                        default=12,
                        type=int,
                        help='the number of epochs for warmup')

    #negtive sampling
    parser.add_argument('--eta', default=5, type=float, help='eta')
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha')

    parser.add_argument('--cutout',
                        default=0.0,
                        type=float,
                        help='cutout factor')

    # random seed
    parser.add_argument('--seed',
                        default=1000,
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

    args = parser.parse_args()

    args.output = args.net + '_outputs'
    args.n_classes = NUM_CLASS[args.dataset_name]
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    args.output = os.path.join(
        args.output, args.dataset_name, '%s' % args.img_size,
        '%s' % args.lb_ratio,
        'warmup_%s_%s_%s_%s_%s_%s' % (args.loss, args.warmup_epochs,
                                   args.cos_margin, args.cos_norm, args.eta, args.alpha))

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
    ema_m = ModelEma(model, args.ema_decay)  # 0.9997

    # Data loading code
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args)
    print("len(lb_train_dataset):", len(lb_train_dataset))
    print("len(ub_train_dataset):", len(ub_train_dataset))
    print("len(val_dataset):", len(val_dataset))

    lb_train_loader = Data.DataLoader(
        lb_train_dataset,
        batch_size=args.warmup_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = Data.DataLoader(val_dataset,
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

    criterion = BAMLoss()

    # centers = torch.zeros(args.n_classes, 256).cuda()
    centers = torch.zeros(args.n_classes, model.in_features).cuda()
    means = torch.zeros(2, args.n_classes).cuda()
    variances = torch.zeros(2, args.n_classes).cuda()
    mean_variances = torch.zeros(args.n_classes).cuda()

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.warmup_epochs):

        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(lb_train_loader, model, ema_m, optimizer, scheduler,
                     epoch, args, logger, criterion, means, variances,
                     mean_variances)

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

        macroF1, microF1, mAP = results_bbam[5], results_bbam[6], results_bbam[3]#11
        macroF1_ema, microF1_ema, mAP_ema = results_bbam_ema[
            5], results_bbam_ema[6], results_bbam_ema[3]#11

        macroF1s.update(macroF1)
        microF1s.update(microF1)
        mAPs.update(mAP)
        macroF1s_ema.update(macroF1_ema)
        microF1s_ema.update(microF1_ema)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.warmup_epochs - epoch - 1))

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

        logger.info("{} | Set best mAP {} in ep {}".format(
            epoch, best_mAP, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(
            best_regular_mAP, best_regular_epoch))

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
            filename=os.path.join(args.output, 'warmup_model.pth.tar'))

        logger.info("{} | Set best mAP {} in ep {}".format(
            epoch, best_mAP, best_epoch))
        logger.info(
            "   | best regular mAP {} best regular MacroF1 {} best regular MicroF1 {} in ep {}"
            .format(best_regular_mAP, best_regular_macroF1,
                    best_regular_microF1, best_regular_epoch))

        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch,
                                               best_regular_epoch) > 4:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info(
                        "epoch - best_epoch = {}, stop!".format(epoch -
                                                                best_epoch))
                    break

    print("Best mAP:", best_mAP)

    np.save(args.output + "/pretrain_results.npy",
            (np.array(results_bbam_all), np.array(results_all)))
    np.save(args.output + "/pretrain_results_ema.npy",
            (np.array(results_bbam_ema_all), np.array(results_ema_all)))

    if summary_writer:
        summary_writer.close()

    return 0


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


def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args,
          logger, criterion, means, variances, mean_variances):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    loss_base = AverageMeter('L_%s' % (args.loss), ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(args.steps_per_epoch,
                             [loss_base, lr, losses, mem],
                             prefix="Epoch: [{}/{}]".format(
                                 epoch, args.warmup_epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, ((inputs_w, inputs_s), targets) in enumerate(train_loader):

        # *********compute loss***************

        batch_size = inputs_w.size(0)

        inputs = torch.cat([inputs_w, inputs_s], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)

        logits_w, logits_s = torch.split(logits[:], batch_size)
        logits_s_bbam, logits_s_bbam1 = applyBBAM(logits_s, targets, means, variances,
                                                  mean_variances, epoch, args)

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
            outputs_bbam = BBAM(outputs, targets, means, variances, mean_variances,
                                -1, args)
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
        "  Macro F1: {}\tMicro F1: {}\tCoverage: {}\tRanking Loss: {}\t"
        "Ranking AP: {}\tHamming Loss: {}\tOne Loss: {}\tmAP: {}\n".format(
            results[5], results[6], results[0], results[1], results[2],
            results[7], results[10], results[3]))#11

    results_bbam = classification_metrics_ml(outputs_bbam_total, targets_total)

    print("Calculating performance after bbam:")
    logger.info(
        "  Macro F1: {}\tMicro F1: {}\tCoverage: {}\tRanking Loss: {}\t"
        "Ranking AP: {}\tHamming Loss: {}\tOne Loss: {}\tmAP: {}\n".format(
            results_bbam[5], results_bbam[6], results_bbam[0], results_bbam[1],
            results_bbam[2], results_bbam[7], results_bbam[10], results_bbam[3]))#11

    return results_bbam, results


if __name__ == '__main__':
    main()
