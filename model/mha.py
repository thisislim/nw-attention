import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, normal_

from model.embedding import TrainableEmbedding
from model.linear import MLP
from utils.edit_distance import DNA_ALPHABET


class MultiheadAttentionMapper(nn.Module):

    def __init__(self, d_model, d_embedding, nhead, seq_max_len=256, d_ff=256, dropout=0.01, activation=F.relu, 
                 readin_layers=1, sa_layers=1, norm=None, device=None, dtype=None):

        super(MultiheadAttentionMapper, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.seq_max_len = seq_max_len

        self.readin = MLP(d_embedding, seq_max_len*d_model, seq_max_len*d_model, readin_layers, dropout=dropout)

        sa_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation, 1e-5, batch_first=True)
        self.self_attn_layers = nn.ModuleList([copy.deepcopy(sa_layer) for _ in range(sa_layers)])

        self.pre_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model) # for tgt
        self.k_proj = nn.Linear(d_model, d_model) # for src

        self.attn_proj = nn.Sequential(
            nn.Linear(self.nhead, self.nhead), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(self.nhead, 1)

        )

        self.softmax = nn.Softmax(dim=-1)

        self.device = device
        
        self._reset_parameters()
        self.readin._reset_parameters()
        

    def forward(self, enc_src, enc_tgt):
        
        batch_size = enc_src.shape[0]

        enc_src = self.readin(enc_src).reshape(batch_size, self.seq_max_len, self.d_model)
        enc_tgt = self.readin(enc_tgt).reshape(batch_size, self.seq_max_len, self.d_model)

        for mod in self.self_attn_layers:
            enc_src = mod(enc_src)

        for mo in self.self_attn_layers:
            enc_tgt = mod(enc_tgt)

        # norm before attention 
        enc_src = self.pre_norm(enc_src)
        enc_tgt = self.pre_norm(enc_tgt)

        enc_tgt = self.q_proj(enc_tgt).reshape(batch_size, self.nhead, self.seq_max_len, self.head_dim)
        enc_src = self.k_proj(enc_src).reshape(batch_size, self.nhead, self.seq_max_len, self.head_dim)

        attention = torch.matmul(enc_tgt, enc_src.permute(0, 1, 3, 2)).permute(0, 2, 3, 1)

        attention = self.attn_proj(attention).squeeze(-1)

        return self.softmax(attention)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        self.attn_proj.apply(self._init_normal)
        
    def _init_normal(self, m):
        if type(m) == nn.Linear:
            normal_(m.weight)
            m.bias.data.zero_()


