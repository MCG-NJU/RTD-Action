# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""Modules to compute the matching cost and solve the corresponding LSAP."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import (generalized_prop_iou, pairwise_temporal_iou,
                          prop_cl_to_se, prop_se_to_cl)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network. For efficiency reasons, the targets don't
    include the no_object. Because of this, in general, there are more
    predictions than targets.

    In the first stage of optimal bipartite matching, we do a 1-to-1 matching
    of the best predictions, while the others are un- matched
    (and thus treated as non-objects).
    In the second stage of relaxed matching, a groundtruth is matched with
    more than one predictions, according to tIoU.

    Args:
        cost_class (float): Relative weight of the classification error
            in the matching cost
        cost_bbox (float): Relative weight of the L1 error of the bounding box
            coordinates in the matching cost
        cost_giou (float): Relative weight of the giou loss
            of the bounding box in the matching cost
        stage (int): Stage ID.
        relax_rule (str): Rule of relaxation. 'thresh' or 'topk'.
        relax_thresh (float): A certain threshold.Predictions with tIoU higher
            than the threshold are marked as positive samples.
        relax_topk (int): Number of top predictions be relaxed. Default: 1.
    """
    def __init__(self,
                 cost_class,
                 cost_bbox,
                 cost_giou,
                 stage,
                 relax_rule,
                 relax_thresh,
                 relax_topk=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'
        self.stage = stage
        self.relax_rule = relax_rule
        self.relax_thresh = relax_thresh
        self.relax_topk = relax_topk

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching.

        Args:
            outputs (dict): A dict that contains at least these entries:
                "pred_logits": Tensor of classification logits.
                    Shape: (batch_size, num_queries, num_classes)
                "pred_boxes": Tensor of the predicted box coordinates.
                    Shape: (batch_size, num_queries, 2)

            targets: (list): A list of targets (len(targets) = batch_size),
            each target is a dict containing:
                "labels": Tensor of class labels.
                    Shape: (num_target_boxes, ) (num_target_boxes is the number of
                    ground-truth objects in the sample)
                "boxes": Tensor of target box coordinates.
                    Shape: (num_target_boxes, 2)

        Returns:
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs['pred_logits'].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(
            -1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(
            0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        if len(tgt_bbox) == 0:
            return None
        # Compute the classification cost.
        # Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, prop_se_to_cl(tgt_bbox), p=1)

        # Compute the giou cost betwen boxes
        # Temporal annotations are already in [start, end] format
        cost_giou = -generalized_prop_iou(prop_cl_to_se(out_bbox), tgt_bbox)

        # Final cost matrix
        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class +
             self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        # Relaxed matching.
        if (self.stage == 2):
            result_indices = []
            for batch_id in range(len(targets)):
                result_indices.append(list(indices[batch_id]))
                gt_boxes = targets[batch_id]['boxes']
                pred_boxes = prop_cl_to_se(outputs['pred_boxes'][batch_id])

                if len(gt_boxes) == 0:
                    tiou = np.zeros((len(pred_boxes), 1))
                    continue
                else:
                    # Calculate pairwise tIoU.
                    # Shape: (num_targets, num_predictions)
                    tiou = pairwise_temporal_iou(
                        pred_boxes.detach().cpu().numpy(),
                        gt_boxes.detach().cpu().numpy())

                if (self.relax_rule == 'thresh'):
                    # Get the max_tiou of each prediction.
                    max_tiou = tiou.max(axis=0)
                    # Get indices of corresponding groundtruths (gts).
                    max_tiou_indices = tiou.argmax(axis=0)
                    # Get indices of predictions, whose max_tiou is higher than the threshold.
                    pred_idx = np.where(max_tiou >= self.relax_thresh)[0]
                    # Get indices of corresponding gts matched with those predictions.
                    gt_idx = max_tiou_indices[pred_idx]

                    # Perform relaxed matching.
                    for i, j in zip(gt_idx, pred_idx):
                        if (j not in result_indices[batch_id][0]):
                            result_indices[batch_id][0] = np.append(
                                result_indices[batch_id][0], j)
                            result_indices[batch_id][1] = np.append(
                                result_indices[batch_id][1], i)

                elif (self.relax_rule == 'topk'):
                    # Get indices of predictions, whose max_tiou is ranked in topK of gts.
                    pred_idx = torch.from_numpy(tiou).argsort(
                        dim=1)[:, -self.relax_topk:].reshape(-1).tolist()
                    for i in range(len(pred_idx)):
                        if (pred_idx[i] not in result_indices[batch_id][0]):
                            result_indices[batch_id][0] = np.append(
                                result_indices[batch_id][0], pred_idx[i])
                            result_indices[batch_id][1] = np.append(
                                result_indices[batch_id][1], i)

            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64))
                    for i, j in result_indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            stage=args.stage,
                            relax_rule=args.relax_rule,
                            relax_thresh=args.relax_thresh)
