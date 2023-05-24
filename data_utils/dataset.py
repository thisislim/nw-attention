import pickle
import torch
import numpy as np
import Levenshtein

from torch.utils.data import Dataset
from utils.edit_distance import string_to_index


class SingleEditDistanceDataset(Dataset):

    def __init__(self, file_path, task=None):
        
        self.data_path = file_path
        self.task = task

        with open(self.data_path, 'rb') as f:
            sequences, distances, nw_matrices = pickle.load(f)
        
        self.sequences = sequences
        self.distances = distances
        self.nw_matrices = nw_matrices

        self.num_sequences = sequences.shape[0]
        self.seq_max_len = sequences.shape[-1]

        self.norm_constant = self.seq_max_len
        
        if self.task in ['ednw']:
            self.softmax = torch.nn.Softmax(dim=-1)
    
    def __len__(self):
        return int(self.num_sequences * (self.num_sequences - 1))

    def __getitem__(self, index):
        # start_time = time.time()
        idx1 = index // (self.num_sequences - 1)
        idx2 = index % (self.num_sequences - 1)
        if idx2 >= idx1:
            idx2 += 1
        
        src = self.sequences[idx1].to(torch.int64)
        tgt = self.sequences[idx2].to(torch.int64)

        # normalized distance
        d = self.distances[idx1, idx2] / self.norm_constant

        # needleman-wunsch matrix
        if self.task in ['ednw']:
            m = self.nw_matrices[idx1, idx2].to(torch.float32)
            m = self.softmax(m)
        
        # no matrix
        elif self.task == 'vanilla':
            m = 0
        
        return src, tgt, d, m

