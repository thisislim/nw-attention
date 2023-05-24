import argparse
import os
import pickle
import json
import random

import numpy as np
import torch

from utils.edit_distance import string_to_index, compute_ed_matrix_multiprocess
from utils.needleman_wunsch import generate_nw_matrices_multiprocess


class EditDistanceDatasetGenerator:
    '''
    save strings into pkl file used for model training.
    '''

    def __init__(self, strings, n_thread=4):
        self.strings = strings
        self.raw_max_len = max([max([len(s) for s in strings[dataset]]) for dataset in strings])

    
    def save_seq_pickle(self, out_dir):
        seq_dir = os.path.dirname(out_dir)
        if seq_dir != '' and not os.path.exists(seq_dir):
            os.makedirs(seq_dir)
        with open(out_dir, 'wb') as f:
            pickle.dump((self.strings, self.raw_max_len), f)

    def save_metadata(self, out_dir):
    
        with open(out_dir, 'rb') as g:
            strings, raw_max_len = pickle.load(g)
        
        meta_dir = os.path.dirname(out_dir)
        if meta_dir != '' and not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
        
        seq_max_len_lst = []

        for dset in strings:
            sequences = np.asarray([string_to_index(s, length=raw_max_len) for s in strings[dset]])
            for s in sequences:
                seq_max_len_lst.append(len(s))

        meta = {
            'seq_max_len' : max(seq_max_len_lst),
        }

        with open(meta_dir + '/' + 'meta.json', 'w') as f:
            json.dump(meta, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default="./data/qiita/", help='Data save path')
    parser.add_argument('--train_size', type=int, default=7000, help='Training sequences')
    parser.add_argument('--val_size', type=int, default=700, help='Validation sequences')
    parser.add_argument('--test_size', type=int, default=1500, help='Test sequences')
    parser.add_argument('--src_dir', type=str, default='./data/qiita.txt', help='Source data path')
    parser.add_argument('--shuffle', type=str, default='False', help='shuffle strings')
    args = parser.parse_args()

    args.shuffle = True if args.shuffle == 'True' else False
    
    # load and divide sequences 
    with open(args.src_dir, 'rb') as f:
        L = f.readlines()

    
    L = [l.decode('utf-8-sig').replace("\n", "") for l in L]
    
    if args.shuffle:
        random.shuffle(L)

    strings = {
        'train': L[:args.train_size],
        'val': L[args.train_size:args.train_size + args.val_size],
        'test': L[args.train_size + args.val_size:args.train_size + args.val_size + args.test_size]
    }

    data = EditDistanceDatasetGenerator(strings=strings)
    data.save_seq_pickle(args.out_dir)
    data.save_metadata(args.out_dir)

