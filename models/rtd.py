# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""RTD-Net model and criterion classes."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized

from .matcher import build_matcher
from .position_embedding import build_position_embedding
from .transformer import build_transformer


class RTD(nn.Module):
    """RTD-Net for temporal action proposal generation (TAPG).

    Args:
        position_embedding (obj): Object of position_embedding.
        transformer (obj): Object of transformer.
        num_classes (int): Number of action classes, only one for TAPG.
        num_queries (int): Number of action queries, the maximal number of proposals
            RTD-Net generates in a sample (32 for THUMOS14).
        stage (int): Stage ID.
        aux_loss (bool): True if auxiliary decoding losses
            (loss at each decoder layer) are to be used. Default: False.
    """
    def __init__(self,
                 position_embedding,
                 transformer,
                 num_classes,
                 num_queries,
                 stage,
                 aux_loss=False):

        super().__init__()
        self.num_queries = num_queries

        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        input_dim = 2048
        self.input_proj = nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=1)

        self.iou_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim,
                      self.hidden_dim * 2,
                      kernel_size=3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim * 2,
                      self.hidden_dim,
                      kernel_size=3,
                      padding=1))
        self.iou_embed = MLP(hidden_dim, hidden_dim * 2, 1, 3)
        self.stage = stage

        self.aux_loss = aux_loss
        self.position_embedding = position_embedding

    def forward(self, locations, samples, s_e_scores):
        """Forward process of RTD-Net.

        Args:
            locations (torch.Tensor): Temporal locations of samples.
                Shape：(batch_size, T, 1).
            samples (torch.Tensor): Features of samples.
                Shape：(batch_size, T, C).
            s_e_scores (torch.Tensor): Predicted start and end score.
                Shape：(batch_size, T, 2).


        Returns:
            out (dict): A dict with the following elements:
                'pred_logits': the classification logits (including no-object) for all queries.
                    Shape: (batch_size, num_queries, (num_classes + 1)).
                'pred_boxes': The normalized boxes coordinates for all queries, represented as
                    (center_x, center_y, height, width). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
                    See PostProcess for information on how to retrieve the unnormalized bounding box.
                    Shape: (batch_size, num_queries, 2).
                'pred_iou': Completeness score of predictions, which measure the overlap
                    between predictions and targets.
                    Shape: (batch_size, num_queries, 1).
                'aux_outputs': Optional, only returned when auxilary losses are activated. It is a list of
                    dictionaries containing the two above keys for each decoder layer.
        """
        bs = s_e_scores.shape[0]

        # boundary-attentive representations
        features_flatten = samples.flatten(0, 1)
        projected_fts = self.input_proj(
            features_flatten.unsqueeze(-1).unsqueeze(-1))
        projected_fts = projected_fts.view((bs, -1, self.hidden_dim // 2))
        scaling_factor = 2
        s = s_e_scores[:, :, 0] * scaling_factor
        e = s_e_scores[:, :, 1] * scaling_factor
        features_s = torch.mul(projected_fts, s.unsqueeze(-1))
        features_e = torch.mul(projected_fts, e.unsqueeze(-1))
        features = torch.cat((features_s, features_e), dim=2)

        pos = self.position_embedding(locations)

        hs = self.transformer(features, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        proposal_fts = hs[-1, :, :, :].permute(0, 2, 1)
        proposal_fts = self.iou_conv(proposal_fts)
        proposal_fts = proposal_fts.permute(0, 2, 1)
        outputs_iou = self.iou_embed(proposal_fts).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_iou': outputs_iou
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                    outputs_coord, outputs_iou)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'pred_logits': a,
            'pred_boxes': b,
            'pred_iou': c
        } for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],
                             outputs_iou[:-1])]


class SetCriterion(nn.Module):
    """This class computes the loss for RTD-Net.

    The process happens in two steps:
    1) We compute hungarian assignment between
        ground truth boxes and the outputs of the model
    2) We supervise each pair of matched
        ground-truth / prediction (supervise class and box)

    Args:
        num_classes (int): Number of action categories,
            omitting the special no-action category.
        matcher (obj): Module able to compute a matching
            between targets and proposals.
        weight_dict (dict): Dict containing as key the names of the losses
            and as values their relative weight.
        eos_coef (float): Relative classification weight
            applied to the no-object category
        losses (list): List of all the losses to be applied.
            See get_loss for list of available losses.
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL) targets dicts must contain the key
        'labels' containing a tensor of dim [nb_target_boxes].

        Args:
            outputs (dict): Dict of RTD outputs.
            targets (list): A list of size batch_size. Each element is a dict composed of:
                'labels': Labels of groundtruth instances (0: action).
                'boxes': Relative temporal ratio of groundtruth instances.
                'video_id': ID of the video sample.
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

        Returns:
            losses (dict): Dict of losses.
        """
        assert 'pred_logits' in outputs
        if indices is None:
            losses = {'loss_ce': 0}
            if log:
                losses['class_error'] = 0
            return losses

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number
        of predicted non-empty boxes This is not really a loss, it is intended
        for logging purposes only.

        It doesn't propagate gradients
        """
        if indices is None:
            losses = {'cardinality_error': 0}
            return losses

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "boxes"
        containing a tensor of dim [nb_target_boxes, 4] The target boxes are
        expected in format (center_x, center_y, w, h), normalized by the image
        size.

        Args:
            outputs (dict): Dict of RTD outputs.
            targets (list): A list of size batch_size. Each element is a dict composed of:
                'labels': Labels of groundtruth instances (0: action).
                'boxes': Relative temporal ratio of groundtruth instances.
                'video_id': ID of the video sample.
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            num_boxes (int): Number of positive samples.

        Returns:
            losses (dict): Dict of losses.
        """
        if indices is None:
            return {'loss_bbox': 0, 'loss_giou': 0}

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes,
                              box_ops.prop_se_to_cl(target_boxes),
                              reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_prop_iou(box_ops.prop_cl_to_se(src_boxes),
                                         target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_iou(self, outputs, targets, indices, num_boxes):
        """tIoU loss for the completeness head.

        Args:
            outputs (dict): Dict of RTD outputs.
            targets (list): A list of size batch_size. Each element is a dict composed of:
                'labels': Labels of groundtruth instances (0: action).
                'boxes': Relative temporal ratio of groundtruth instances.
                'video_id': ID of the video sample.

        Returns:
            losses (dict): Dict of losses.
        """
        assert 'pred_iou' in outputs
        assert 'pred_boxes' in outputs

        bs = outputs['pred_boxes'].shape[0]

        pred_boxes = outputs['pred_boxes']
        preds_iou = outputs['pred_iou']

        tgt_iou = []
        for i in range(bs):
            pred_boxes_per_seg = pred_boxes[i, :, :]
            target_boxes_per_seg = targets[i]['boxes']
            if len(target_boxes_per_seg) == 0:
                tiou = torch.zeros(
                    (len(pred_boxes_per_seg))).to(pred_boxes_per_seg.device)
            else:
                tiou = box_ops.generalized_prop_iou(
                    box_ops.prop_cl_to_se(pred_boxes_per_seg),
                    target_boxes_per_seg)
                tiou = torch.max(tiou, dim=1)[0]
            tgt_iou.append(tiou)

        tgt_iou = torch.stack(tgt_iou, dim=0).view(-1)
        preds_iou = preds_iou.view(-1)

        # We take target iou larger than 0.7 as positive samples.
        pos_ind = torch.nonzero(tgt_iou > 0.7)
        m_ind = torch.nonzero((tgt_iou <= 0.7) & (
            tgt_iou > 0.3)).squeeze().cpu().detach().numpy()
        neg_ind = torch.nonzero(tgt_iou < 0.3).squeeze().cpu().detach().numpy()

        # We take all positive samples.
        # To balance the number of different kinds of samples,
        # we randomly sample len(pos_ind) middle samples,
        # and 2*len(pos_ind) negative samples as training samples.
        sampled_m_ind = np.random.choice(m_ind, len(pos_ind))
        sampled_neg_ind = np.random.choice(neg_ind, 2 * len(pos_ind))

        iou_mask = (tgt_iou > 0.7).float()
        iou_mask[sampled_m_ind] = 1.
        iou_mask[sampled_neg_ind] = 1.
        iou_loss = F.smooth_l1_loss(preds_iou, tgt_iou.squeeze()).float()
        iou_loss = torch.sum(
            iou_loss * iou_mask) / (1e-6 + torch.sum(iou_mask)).float()

        losses = {'loss_iou': iou_loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        '''
        Args:
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

        Returns:
            A tuple composed of two tensors:
                the first is batch idx,
                the second is sample_idx.
        '''
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'iou': self.loss_iou
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """Loss computation.

        Args:
            outputs (dict): Dict of RTD outputs, which are tensors.
            targets (dict): List of dicts, such that len(targets) == batch_size.
                The expected keys in each dict depends on the losses applied.

        Returns:
            losses (dict): Dict of losses.
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs and 'iou' not in self.losses:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the
    THUMOS14 dataset evaluation tool."""
    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size
        self.interval = args.interval
        self.absolute_position = args.absolute_position
        self.stage = args.stage

    @torch.no_grad()
    def forward(self, outputs, num_frames, base):
        """ Perform the computation
        Parameters:
            outputs (dict): Dict of RTD outputs.
            num_frames (torch.Tensor): Number of frames in samples.
                Shape: (batch_size, )
            base (torch.Tensor): Index of the base/first frame in samples.
                Shape: (batch_size, )
            raw outputs of the model
            num_frames: tensor of dimension [batch_size x 1] containing the frame duration of each videos of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_iou = outputs['pred_logits'], outputs[
            'pred_boxes'], outputs['pred_iou']

        assert len(out_logits) == len(num_frames)
        num_frames = num_frames.reshape((len(out_logits), 1))

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.prop_cl_to_se(out_bbox)
        if not self.absolute_position:
            boxes = box_ops.prop_relative_to_absolute(boxes, base,
                                                      self.window_size,
                                                      self.interval)
        # and from relative [0, 1] to absolute [0, height] coordinates
        else:
            bs, ws, _ = boxes.shape
            scale_fct = num_frames.unsqueeze(-1).repeat(
                (1, ws, 2)).to(boxes.device)

            boxes = boxes * scale_fct

        if self.stage != 3:
            results = [{
                'scores': s,
                'labels': l,
                'boxes': b,
                'iou_score': i
            } for s, i, l, b in zip(scores, out_iou, labels, boxes)]
        if self.stage == 3:
            new_scores = 0.5 * (scores.squeeze() + out_iou.squeeze())
            results = [{
                'scores': s,
                'labels': l,
                'boxes': b,
                'iou_score': i
            } for s, i, l, b in zip(new_scores, out_iou, labels, boxes)]

        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 1
    device = torch.device(args.device)

    position_embedding = build_position_embedding(args)

    transformer = build_transformer(args)

    model = RTD(position_embedding=position_embedding,
                transformer=transformer,
                num_classes=num_classes,
                num_queries=args.num_queries,
                stage=args.stage,
                aux_loss=args.aux_loss)

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': args.bbox_loss_coef,
        'loss_iou': args.iou_loss_coef
    }
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v
                 for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.stage != 3:
        losses = ['labels', 'boxes', 'cardinality']
    else:
        losses = ['iou']

    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=args.eos_coef,
                             losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, postprocessors
