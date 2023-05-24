import os
import pickle
import json
import random

import torch

from .dataset import SingleEditDistanceDataset


def load_dataset_list(root_dir):
    '''
    return [train/val/test] dictionary of dataset directory list
    '''
    dir_list = [root_dir + '/train', root_dir + '/val', root_dir + '/test']

    dataset_dir_list = {}

    for subdir in dir_list:
        key = subdir.split('/')[-1]
        dataset_dir_list[key] = []

        file_list = os.listdir(subdir)
        dir_list = []
        for fln in file_list:
            dir_list.append(subdir + '/' + fln)
        random.shuffle(dir_list)
        dataset_dir_list[key] = dir_list
    return dataset_dir_list
    
def load_dataset(dataset_dir, task):        
    return SingleEditDistanceDataset(dataset_dir, task)

def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):

    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size,
                                         shuffle=shuffle,  
                                         num_workers=num_workers)
    return loader

def load_metadata(root_dir):
    with open(root_dir + '/meta.json', 'r') as f:
        meta = json.load(f)
    return meta


