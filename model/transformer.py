import warnings
import copy

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from torch.nn import (TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder)
from model.embedding import TrainableEmbedding
from model.linear import MLP
from utils.distance_functions import DISTANCE_TORCH
from utils.edit_distance import DNA_ALPHABET


N_TOKENS = len(DNA_ALPHABET)


class AttentionTransformerEncoder(nn.Module):

    def __init__(self, d_model, d_embedding, nhead, seq_max_len=256, dim_feedforward=1024, dropout=0.1, activation=F.relu, 
                 layer_norm_eps=1e-5, enc_layers=6, readout_layers=2, norm=None, batch_first=True, norm_first=False, segment_size=None, 
                 device=None, dtype=None):
        super(AttentionTransformerEncoder, self,).__init__()

        if segment_size is not None:
            self.segment_size = segment_size
            self.padding = (-seq_max_len) % segment_size
            seq_max_len += self.padding
            seq_max_len = seq_max_len // self.segment_size
        else:
            self.segment_size = 1
            self.padding = 0
    
        self.embedding = TrainableEmbedding(d_model, seq_max_len, self.segment_size, self.padding, dropout, device, dtype)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, 
                                                   layer_norm_eps, batch_first, norm_first, device, dtype)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(enc_layers)])

        
        if readout_layers > 0:
            self.readout = MLP(dim_in=seq_max_len*d_model, dim_hidden=d_embedding, dim_out=d_embedding, 
                               num_layers=readout_layers, dropout=dropout)
        else: self.readout = None
            
        self.readout_layers = readout_layers

        self.norm = nn.LayerNorm(d_model, eps=1e-6) if norm else None
        self.device = device
        self.batch_first = batch_first

    def forward(self, src, mask=None, src_key_padding_mask=None):

        src = self.embedding(src)
        
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        if self.readout is not None:
            output = self.readout(output)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def generate_pad_mask(seq):
    mask = (seq == DNA_ALPHABET["<pad>"])
    return mask


