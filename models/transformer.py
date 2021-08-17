# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""RTD-Net Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Transformer(nn.Module):
    """Implementation of transformer.

    Args:
        d_model (int): Hidden dimension for layers of model.
            Default: 512.
        nhead (int): Number of attention heads.
            Default: 8.
        num_encoder_layers (int): Number of MLP encoder layers.
            Default: 3.
        num_decoder_layers (int): Number of transformer decoder layers.
            Default: 6.
        dim_feedforward (int): Hidden dimension for FFNs.
            Default: 2048.
        dropout (float): Dropout ratio.
            Default: 0.1.
        activation (str): Activation function.
            Default: 'relu'
        normalize_before (bool): Whether to perform layer normalization
            before attention. Default: False.
        return_intermediate_dec (bool): Whether to return intermediate
            representations. Default: False
    """
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        num_encoder_layers = 3
        print('Use {}-layer MLP as encoder'.format(num_encoder_layers))
        self.encoder = simpleMLP(d_model * 2, d_model, d_model,
                                 num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        """Forward of transformer.

        Args:
            src (torch.Tensor): Src features of samples.
                Shape: (batch_size, T, C)
            query_embed (torch.Tensor): Query embedding for decoder.
                Shape: (num_query, C)
            pos_embed (torch.Tensor): Positional embedding for encoder and
                decoder, with the same shape as `src`.
                Shape: (batch_size, T, C)

        Returns:
            decoder_output with shape (num_query, num_classes + 1)
            encoder_output with shape (batch_size, T, C)
        """
        # NxTxC to TxNxC
        bs, t, c = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)

        src = torch.cat([src, pos_embed], dim=2)
        memory = self.encoder(src)

        hs = self.decoder(tgt,
                          memory,
                          memory_key_padding_mask=None,
                          pos=pos_embed,
                          query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, t)


class simpleMLP(nn.Module):
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


class TransformerDecoder(nn.Module):
    """Implementation of transformer decoder.

    Args:
        decoder_layer (obj): Object of TransformerDecoderLayer.
        num_layers (int): Number of decoder layers.
        norm (obj): Object of normalization. Default: None.
        return_intermediate (bool): Whether to return intermediate
            representations. Default: False
    """
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """Forward of transformer decoder.
        Args:
            tgt (torch.Tensor): Action queries of transformer decoder.
                Shape: (num_query, batch_size, C)
            memory (torch.Tensor): Output of MLP encoder.
                Shape: (T, batch_size, C)

        Returns:
            Output of each layer / the last layer of the transformer decoder.
        """
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    """Implementation of decoder layer in transformer.

    Args:
        d_model (int): Hidden dimension for layers of model.
            Default: 512.
        nhead (int): Number of attention heads.
            Default: 8.
        dim_feedforward (int): Hidden dimension for FFNs.
            Default: 2048.
        dropout (float): Dropout ratio.
            Default: 0.1.
        activation (str): Activation function.
            Default: 'relu'
        normalize_before (bool): Whether to perform layer normalization
            before attention. Default: False.
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """Forward of decoder layer in transformer layer normalization
        performed after attention.

        Args:
            tgt (torch.Tensor): Action queries of transformer decoder.
                Shape: (num_query, batch_size, C)
            memory (torch.Tensor): Output of MLP encoder.
                Shape: (T, batch_size, C)
            pos_embed (torch.Tensor): Positional embedding for encoder and
                decoder, with the same shape as `memory`.
                Shape: (T, batch_size, C)
            query_pos (torch.Tensor): Query embedding for decoder,
                with the same shape as `tgt`.
                Shape: (num_query, batch_size, C)

        Returns:
            tgt after self-attention and cross-attention.
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """Forward of decoder layer in transformer, layer normalization
        performed before attention.

        Similar to 'forward_pos'.
        """
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(d_model=args.hidden_dim,
                       dropout=args.dropout,
                       nhead=args.nheads,
                       dim_feedforward=args.dim_feedforward,
                       num_encoder_layers=args.enc_layers,
                       num_decoder_layers=args.dec_layers,
                       normalize_before=args.pre_norm,
                       return_intermediate_dec=True)


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')
