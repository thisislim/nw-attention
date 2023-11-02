import os
import argparse

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='data file path')
    parser.add_argument('--out', type=str, default='', help='out file path')

    args = parser.parse_args()
    data_dir = os.path.dirname(args.data)

    with open(args.data, 'r') as f:
        fastq = f.readlines()

    N = len(fastq)
    reads = []

    for i in tqdm(range(1, N, 4), desc='sorting reads'):
        if 'N' not in fastq[i]:
            reads.append(fastq[i].replace('\n', ''))

    with open(args.out, 'w') as g:
        for r in tqdm(reads, desc='writing reads'):
            g.write(r + '\n')

    
