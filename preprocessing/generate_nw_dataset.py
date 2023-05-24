import argparse
import os
import pickle
import random

import numpy as np
import torch
import Levenshtein

from utils.edit_distance import string_to_index, compute_ed_matrix_multiprocess
from utils.needleman_wunsch import generate_nw_matrices_multiprocess


def save_pickle(filename, strings, seq_max_len, index, gap=350, n_thread=4, args=None):
    # slice strings
    string_segments = strings[index*gap:(index+1)*gap]
    # convert strings to indices
    sequences = torch.from_numpy(np.asarray([string_to_index(s, length=seq_max_len) for s in string_segments]))
    # compute edit distance
    distances = torch.from_numpy(compute_ed_matrix_multiprocess(string_segments, string_segments, nthreads=n_thread))
    # generate nw matrices
    nw_matrices = torch.from_numpy(generate_nw_matrices_multiprocess(string_segments, string_segments, nthreads=n_thread)).to(torch.int16)

    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump((sequences, distances, nw_matrices), f)


def save_attn(out_dir, strings, seq_max_len, num_attn_seq):

    attn_distances = [2**i * 5 for i in range(0, num_attn_seq)]

    attn_strings = [strings[random.randint(0, len(strings)-1)]]
    
    for attn_d in attn_distances:
        i, d = 0, 0
        while d != attn_d:
            tgt = strings[i]
            d = Levenshtein.distance(attn_strings[0], tgt)
            i += 1

        attn_strings.append(tgt)

    sequences = torch.from_numpy(np.asarray([string_to_index(s, length=seq_max_len) for s in attn_strings]))
    distances = torch.from_numpy(compute_ed_matrix_multiprocess(attn_strings, attn_strings, 2))
    nw_matrices = torch.from_numpy(generate_nw_matrices_multiprocess(attn_strings, attn_strings, 2)).to(torch.int16)

    with open(out_dir, 'wb') as h:
        pickle.dump((sequences, distances, nw_matrices), h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default="./data/qiita", help='Data save path')
    parser.add_argument('--src_dir', type=str, default='./data/qiita/seq.pkl', help='Source data path')
    parser.add_argument('--key', type=str, default=None, help='Dataset dictionary key')
    parser.add_argument('--index', type=int, default=0, help='Dataset string index')
    parser.add_argument('--gap', type=int, default=350, help='Gap for string segments')
    parser.add_argument('--num_attn_data', type=int, default=0, help='Generate Attention val data')
    args = parser.parse_args()
    

    with open(args.src_dir, 'rb') as f:
        strings, max_len = pickle.load(f)

    strings = strings[args.key]

    dset_dir = os.path.dirname(args.out_dir+'/'+f'{args.key}/')
    if dset_dir != '' and not os.path.exists(dset_dir):
        os.makedirs(dset_dir)

    filename = f'{dset_dir}/{args.key}_{args.index}'
    
    
    print(f'Generating {filename}')
    save_pickle(filename, strings, max_len, args.index, args.gap)

    if args.num_attn_data > 0:
        attn_fln = f'{args.out_dir}/attn.pkl'
        print(f'generating {attn_fln}')
        save_attn(attn_fln, strings, max_len, args.num_attn_data)

