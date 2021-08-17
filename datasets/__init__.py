# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .thumos14 import build as build_thumos14


def build_dataset(image_set, args):
    if args.dataset_file == 'thumos14':
        return build_thumos14(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
