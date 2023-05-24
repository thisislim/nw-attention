from multiprocessing import Pool
from tqdm import tqdm

import torch
import numpy as np
from utils.nw_align import nw_align


def generate_nw_matrices(A, B):
    return np.array([[nw_align(a, b) for b in B] for a in A])

def generate_nw_matrices_row(args):
    a, B = args
    return [nw_align(a, b) for b in B]

def generate_nw_matrices_multiprocess(A, B, nthreads):
    with Pool(nthreads) as p:
        nw_matrices = list(
            tqdm(
                p.imap(generate_nw_matrices_row, zip(A, [B for _ in A])), 
                total=len(A), 
                desc="Needleman Wunsh matrix {}x{}".format(len(A), len(B))
            )
        )
        return np.array(nw_matrices, np.int16)

def generate_nw_matrices_mp_mute(A, B, nthreads):
    with Pool(nthreads) as p:
        nw_matrices = list(
            p.imap(generate_nw_matrices_row, zip(A, [B for _ in A]))
        )
    return np.array(nw_matrices, np.int16)

