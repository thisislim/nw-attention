import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distance_functions import DISTANCE_TORCH
from utils.edit_distance import DNA_ALPHABET


class PairEncoding(nn.Module):

    def __init__(self, model, distance="hyperbolic", scaling=False):
        super(PairEncoding, self).__init__()

        self.encoder = model
        self.distance_fn = DISTANCE_TORCH[distance]
        self.distance = distance

        self.scaling = None
        if scaling:
            self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
            self.scaling = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        if distance == 'square':
            self.sq_rescale = 1/1.4142135623730951

    def normalize_embedding(self, enc):
        min_scale = 1e-7

        if self.distance == 'hyperbolic':
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10

        return F.normalize(enc, p=2, dim=-1) * self.radius.clamp_min(min_scale).clamp_max(max_scale)

    
    def encode(self, seq, seq_pad_mask=None):
        enc_seq = self.encoder(seq, src_key_padding_mask=seq_pad_mask)
        if self.scaling is not None:
            enc_seq = self.normalize_embedding(enc_seq)

        return enc_seq


    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        batch_size = src.shape[0]

        enc_src = self.encode(src, src_pad_mask)
        enc_tgt = self.encode(tgt, tgt_pad_mask)

        if self.distance == 'square':
            enc_src = enc_src * self.sq_rescale
            enc_tgt = enc_tgt * self.sq_rescale

        enc_distance = self.distance_fn(enc_src.reshape(batch_size, -1), enc_tgt.reshape(batch_size, -1))
        if self.scaling is not None:
            enc_distance = enc_distance * self.scaling

        return enc_src, enc_tgt, enc_distance

        