# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets.thumos14 as thumos
import util.misc as utils
from datasets import build_dataset
from datasets.thumos14_eval import eval_props
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument(
        '--frozen_weights',
        type=str,
        default=None,
        help='Path to the pretrained model, only the mask head will be trained'
    )
    # * Backbone
    parser.add_argument('--backbone',
                        default='resnet50',
                        type=str,
                        help='Name of the convolutional backbone to use')
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help='Type of positional embedding to use on top of the image features'
    )

    # * Transformer
    parser.add_argument('--enc_layers',
                        default=3,
                        type=int,
                        help='Number of encoding layers in the transformer')
    parser.add_argument('--dec_layers',
                        default=6,
                        type=int,
                        help='Number of decoding layers in the transformer')
    parser.add_argument(
        '--dim_feedforward',
        default=2048,
        type=int,
        help='Intermediate size of the FFN in the transformer blocks')
    parser.add_argument(
        '--hidden_dim',
        default=512,
        type=int,
        help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help='Dropout applied in the transformer')
    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries',
                        default=100,
                        type=int,
                        help='Number of query slots')
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument(
        '--no_aux_loss',
        dest='aux_loss',
        action='store_false',
        help='Disables auxiliary decoding losses (loss at each layer)')
    # * Matcher
    parser.add_argument('--set_cost_class',
                        default=1,
                        type=float,
                        help='Class coefficient in the matching cost')
    parser.add_argument('--set_cost_bbox',
                        default=5,
                        type=float,
                        help='L1 box coefficient in the matching cost')
    parser.add_argument('--set_cost_giou',
                        default=2,
                        type=float,
                        help='giou box coefficient in the matching cost')
    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--iou_loss_coef', default=100, type=float)
    parser.add_argument(
        '--eos_coef',
        default=0.1,
        type=float,
        help='Relative classification weight of the no-object class')
    parser.add_argument('--relax_rule', default='topk', type=str)
    parser.add_argument('--relax_thresh', default=0.7, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='thumos14')
    parser.add_argument('--window_size', default=100, type=int)
    parser.add_argument('--interval', default=5, type=int)
    parser.add_argument('--gt_size', default=100, type=int)
    parser.add_argument('--feature_path',
                        default='data/I3D_features',
                        type=str)
    parser.add_argument('--tem_path', default='data/TEM_scores', type=str)
    parser.add_argument('--annotation_path',
                        default='datasets/thumos14_anno_action.json',
                        type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--point_prob_normalize', action='store_true')
    parser.add_argument('--absolute_position', action='store_true')
    parser.add_argument('--stage',
                        default=1,
                        type=int,
                        help='stage-id of RTD-Net')

    parser.add_argument('--output_dir',
                        default='outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load', default='', help='load checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    return parser


def draw(split, loss_dict, axs, epoch, color):
    keys = loss_dict.keys()
    for k in keys:
        axs[k].set_title(k)
        if epoch == 0:
            axs[k].plot(loss_dict[k], color=color, label=k)
        else:
            axs[k].plot(loss_dict[k], color=color)
        plt.pause(0.001)


def draw_stats(axes, stats, epoch, colordict):

    for k, v in stats.items():
        if len(epoch) == 1:
            axes.plot(v, color=colordict[k], label=k)
        else:
            axes.plot(v, color=colordict[k])
    plt.pause(0.001)


def main(args):
    utils.init_distributed_mode(args)
    print('git:\n  {}\n'.format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)
    print(device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    if args.stage == 1:
        for name, value in model_without_ddp.named_parameters():
            if 'iou' in name:
                value.requires_grad = False
        learned_params = filter(lambda p: p.requires_grad,
                                model_without_ddp.parameters())
    elif args.stage == 2:
        for name, value in model_without_ddp.named_parameters():
            if 'class_embed' not in name:
                value.requires_grad = False
        head_params = filter(lambda p: p.requires_grad,
                             model_without_ddp.parameters())
        learned_params = list(head_params)
    else:
        for name, value in model_without_ddp.named_parameters():
            if 'iou' not in name:
                value.requires_grad = False
        head_params = filter(lambda p: p.requires_grad,
                             model_without_ddp.parameters())
        learned_params = list(head_params)

    optimizer = torch.optim.AdamW(learned_params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                        args.batch_size,
                                                        drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=thumos.collate_fn,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=thumos.collate_fn,
                                 num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.rtd.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            pretrained_dict = checkpoint['model']
            # only resume part of model parameter
            model_dict = model_without_ddp.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            model_without_ddp.load_state_dict(model_dict)
            # main_model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded '{}' (epoch {})".format(args.resume,
                                                      checkpoint['epoch'])))

    if args.load:
        checkpoint = torch.load(args.load, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        evaluator, eval_loss_dict = evaluate(model, criterion, postprocessors,
                                             data_loader_val, device, args)
        res = evaluator.summarize()

        test_stats, results_pd = eval_props(res)
        print('test_stats', test_stats)

        if args.output_dir:
            results_pd.to_csv(args.output_dir + 'results_eval.csv')
        return

    print('Start training')
    start_time = time.time()

    fig1 = plt.figure('train', figsize=(18.5, 10.5))
    ax1_train = fig1.add_subplot(231)
    ax2_train = fig1.add_subplot(232)
    ax3_train = fig1.add_subplot(233)
    ax4_train = fig1.add_subplot(234)
    ax5_train = fig1.add_subplot(235)
    ax6_train = fig1.add_subplot(236)

    axs_train = {
        'loss_ce': ax1_train,
        'loss_bbox': ax2_train,
        'loss_giou': ax3_train,
        'cardinality_error': ax4_train,
        'class_error': ax5_train,
        'loss_iou': ax6_train
    }

    fig2 = plt.figure('eval', figsize=(18.5, 10.5))
    ax1_eval = fig2.add_subplot(231)
    ax2_eval = fig2.add_subplot(232)
    ax3_eval = fig2.add_subplot(233)
    ax4_eval = fig2.add_subplot(234)
    ax5_eval = fig2.add_subplot(235)
    ax6_eval = fig2.add_subplot(236)
    axs_eval = {
        'loss_ce': ax1_eval,
        'loss_bbox': ax2_eval,
        'loss_giou': ax3_eval,
        'cardinality_error': ax4_eval,
        'class_error': ax5_eval,
        'loss_iou': ax6_eval
    }

    colordict = {
        '50': 'g',
        '100': 'b',
        '200': 'purple',
        '500': 'orange',
        '1000': 'brown'
    }

    fig3 = plt.figure('test_AR')
    axs_test = fig3.add_subplot(111)

    epoch_list = []
    train_loss_list = {}
    eval_loss_list = {}
    test_stats_list = {}
    best_ar50 = 0
    best_sum_ar = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats, train_loss_dict = train_one_epoch(model, criterion,
                                                       data_loader_train,
                                                       optimizer, device,
                                                       epoch, args)

        for key, value in train_loss_dict.items():
            if key in [
                    'loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error',
                    'class_error', 'loss_iou'
            ]:
                try:
                    train_loss_list[key].append(value.mean())
                except KeyError:
                    train_loss_list[key] = [value.mean()]

        lr_scheduler.step()
        if epoch % 50 == 0 and args.output_dir:
            checkpoint_path = output_dir / 'checkpoint_epoch{}.pth'.format(
                epoch)
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir /
                                        f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        evaluator, eval_loss_dict = evaluate(model, criterion, postprocessors,
                                             data_loader_val, device, args)
        res = evaluator.summarize()

        test_stats, results_pd = eval_props(res)
        for k, v in test_stats.items():
            try:
                test_stats_list[k].append(float(v) * 100)
            except KeyError:
                test_stats_list[k] = [float(v) * 100]

        for key, value in eval_loss_dict.items():
            if key in [
                    'loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error',
                    'class_error', 'loss_iou'
            ]:
                try:
                    eval_loss_list[key].append(value.mean())
                except KeyError:
                    eval_loss_list[key] = [value.mean()]

        print('test_stats', test_stats)

        # debug
        # if args.output_dir:
        #     results_pd.to_csv(args.output_dir+'results_epoch_{}.csv'.format(epoch))

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_AR@{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }
        if (float(test_stats['50']) > best_ar50):
            best_ar50 = float(test_stats['50'])
            with (output_dir / 'log_best_ar50.txt').open('w') as f:
                f.write(json.dumps(log_stats) + '\n')
            checkpoint_path = output_dir / 'checkpoint_best_ar50.pth'
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        current_sum_ar = float(test_stats['50']) + float(
            test_stats['100']) + float(test_stats['200'])
        if (current_sum_ar > best_sum_ar):
            best_sum_ar = current_sum_ar
            with (output_dir / 'log_best_sum_ar.txt').open('w') as f:
                f.write(json.dumps(log_stats) + '\n')
            checkpoint_path = output_dir / 'checkpoint_best_sum_ar.pth'
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.output_dir and utils.is_main_process():
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
        epoch_list.append(epoch)
        if epoch % 2 == 0:
            # split, loss_dict, axs, epoch, color_dict

            draw_stats(axs_test, test_stats_list, epoch_list, colordict)
            axs_test.legend()
            draw('train', train_loss_list, axs_train, epoch, 'b')
            draw('eval', eval_loss_list, axs_eval, epoch, 'g')
            fig1.savefig('train_loss_curve.jpg', dpi=300)
            fig2.savefig('eval_loss_curve.jpg', dpi=300)
            fig3.savefig('test_ar.jpg')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RTD-Net training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
