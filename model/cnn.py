import torch
import torch.nn as nn
import torch.nn.functional as F

from model.linear import MLP

from utils.edit_distance import DNA_ALPHABET



class CNNed(nn.Module):

    def __init__(self, seq_len, d_embedding, channels, kernel_size, stride, layers, pooling='avg', readout_layers=1, dropout=0.0, 
                 activation=None, batch_norm=False, device=None):
        super(CNNed, self).__init__()

        self.layers = layers
        self.kernel_size = kernel_size
        self.vocab_size = len(DNA_ALPHABET)
        self.embedding = nn.Linear(self.vocab_size, channels)

        self.conv = nn.ModuleList()
        for l in range(self.layers):
            self.conv.append(
                nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, 
                          padding=kernel_size // 2, stride=stride)
            )
            seq_len = (seq_len - 1) // stride + 1
            if batch_norm:
                self.conv.append(nn.BatchNorm1d(num_features=channels))
            if activation is not None:
                self.conv.append(nn.ReLU())
            if pooling == 'avg':
                self.conv.append(nn.AvgPool1d(2))
                seq_len //= 2
            elif pooling == 'avg':
                self.conv.append(nn.MaxPool1d(2))
                seq_len //= 2
            
        flat_sz = channels * seq_len
        self.readout = MLP(dim_in=flat_sz, dim_hidden=flat_sz, dim_out=d_embedding, num_layers=readout_layers, dropout=dropout)

    def forward(self, seq, src_key_padding_mask=None):
        # src_key_padding_mask arg for PairEncoder compatibility
        bsz = seq.shape[0]
        
        seq = F.one_hot(seq, self.vocab_size)
        
        seq = self.embedding(seq.to(torch.float32))

        enc_seq = seq.permute(0, 2, 1)
        for mod in self.conv:
            enc_seq = mod(enc_seq)

        enc_seq = enc_seq.reshape(bsz, -1)
        out = self.readout(enc_seq)

        return out

