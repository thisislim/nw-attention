import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.edit_distance import DNA_ALPHABET


N_TOKENS = len(DNA_ALPHABET)


class TrainableEmbedding(nn.Module):
    def __init__(self, embedding_dim, seq_max_len=256, segment_size=1, pad=0, dropout=0.1, device=None, dtype=None):
        
        super(TrainableEmbedding, self).__init__()
        
        self.d_model = embedding_dim
        self.seq_max_len = seq_max_len
        self.pad = pad

        self.linear = nn.Linear(N_TOKENS*segment_size, self.d_model)

        self.pos_embedding = nn.Embedding(seq_max_len, self.d_model, device=device, dtype=dtype)

        self.scale = torch.sqrt(torch.FloatTensor([self.d_model])).to(device)

        self.dropout = nn.Dropout(dropout)

        self.device = device
        

    def forward(self, seq):
        
        batch_size, seq_len = seq.shape
            
        encoding = F.one_hot(seq, N_TOKENS)
        if self.pad > 0:
            encoding = F.pad(encoding, (0, 0, 0, self.pad))

        if seq_len != self.seq_max_len:
            encoding = encoding.reshape(batch_size, self.seq_max_len, -1)

        encoding = self.linear(encoding.to(torch.float32))
        
        pos = torch.arange(0, self.seq_max_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        seq = encoding * self.scale + self.pos_embedding(pos)
        return self.dropout(seq)


