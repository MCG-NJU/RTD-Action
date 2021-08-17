# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""Various positional encodings for the transformer."""
import numpy as np
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    def __init__(self, hidden_dim=256, num_locations=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_locations = num_locations
        self.register_buffer('position_table', self._get_encoding_table())

    def _get_position_angle_vector(self, position):
        return [
            position / np.power(100, 2 * (i // 2) / self.hidden_dim)
            for i in range(self.hidden_dim)
        ]

    def _get_encoding_table(self):
        # TODO: make it with torch instead of numpy
        table = np.array([
            self._get_position_angle_vector(position)
            for position in range(self.num_locations)
        ])
        table[:, 0::2] = np.sin(table[:, 0::2])  # dim 2i
        table[:, 1::2] = np.cos(table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(table).unsqueeze(0)

    def forward(self, locations):
        result = self.position_table[:locations.shape[1]].clone().detach(
        ).repeat(locations.shape[0], 1, 1)
        return result


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""
    def __init__(self, hidden_dim=256, num_locations=100):
        super().__init__()
        self.embedding = nn.Embedding(num_locations, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight)

    def forward(self, locations):
        i = torch.arange(locations.shape[1], device=locations.device)
        embedding = self.embedding(i)

        return embedding.unsqueeze(0).repeat(locations.shape[0], 1, 1)


def build_position_embedding(args):
    # TODO: Pass args.hidden_dim and args.embedding_method
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(
            hidden_dim=args.hidden_dim, num_locations=args.window_size)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(
            hidden_dim=args.hidden_dim, num_locations=args.window_size)
    else:
        raise ValueError(f'not supported {args.position_embedding}')
    return position_embedding


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help='Type of positional embedding to use on top of the image features'
    )
    parser.add_argument('--window_size', default=32, type=int)
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int,
        help='Size of the embeddings (dimension of the transformer)')

    return parser


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('RTD-Net training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    position_embedding = build_position_embedding(args)
    locations = torch.ones(1, args.window_size, 1)
    locations[0, :, 0] = torch.arange(args.window_size)
    position_code = position_embedding(locations)
