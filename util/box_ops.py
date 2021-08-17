# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""Utilities for bounding box manipulation and GIoU."""
import numpy as np
import torch


# rewrite for temporal localization setting
def prop_cl_to_se(x):
    c, l = x.unbind(-1)
    b = [(c - 0.5 * l), (c + 0.5 * l)]
    return torch.stack(b, dim=-1).clamp(0, 1)


def prop_se_to_cl(x):
    s, e = x.unbind(-1)
    b = [(s + e) / 2, (e - s)]
    return torch.stack(b, dim=-1)


def prop_relative_to_absolute(x, base, window_size, interval):
    s, e = x.unbind(-1)
    num_samples = s.shape[1]
    base = base.unsqueeze(1).repeat(1, num_samples).cuda()
    b = [s * window_size * interval + base, e * window_size * interval + base]
    return torch.stack(b, dim=-1)


def segment_tiou(box_a, box_b):
    # gt: [N, 2], detections: [M, 2]
    N = box_a.shape[0]
    M = box_b.shape[0]

    tiou = torch.zeros((N, M)).to(box_a.device)
    for i in range(N):
        inter_max_xy = torch.min(box_a[i, 1], box_b[:, 1])
        inter_min_xy = torch.max(box_a[i, 0], box_b[:, 0])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

        # calculate union
        union = (box_b[:, 1] - box_b[:, 0]) + (box_a[i, 1] -
                                               box_a[i, 0]) - inter

        tiou[i, :] = inter / union

    return tiou  # (N, M)


def pairwise_temporal_iou(candidate_segments, target_segments):
    """Compute intersection over union between segments.
    Args:
        candidate_segments (np.ndarray): 1-dim/2-dim array in format
            [init, end]/[m x 2:=[init, end]].
        target_segments (np.ndarray): 2-dim array in format
            [n x 2:=[init, end]].
    Returns:
        t_iou (np.ndarray): 1-dim array [n] /
            2-dim array [n x m] with IoU ratio.
    """
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    n, m = target_segments.shape[0], candidate_segments.shape[0]
    t_iou = np.empty((n, m), dtype=np.float32)
    for i in range(m):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        t_iou[:, i] = (segments_intersection.astype(float) / segments_union)

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)

    return t_iou


def generalized_prop_iou(props1, props2):
    """rewritten Generalized IoU from https://giou.stanford.edu/ to work under
    temporal localization setting.

    The props should be in [start, end] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    return segment_tiou(props1, props2)
