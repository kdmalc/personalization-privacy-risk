# PFLNIID

import numpy as np
import os
import torch
import pickle
from utils.emg_dataset_class import *


def read_data(dataset, idx, is_train=True, condition_number=1, test_split=0.2, test_split_each_update=False):
    # KAI'S EDITED VERSION WHICH IS NOW USED IN THE CODE
    print(f"Client{idx}, train={is_train}: read_data() called!")
    
    if test_split_each_update:
        raise("Not supported yet.  Idk if this is necessary :/")
    
    all_user_keys = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']
    if dataset.upper()=='CPHS':
        with open(r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\Data\continuous_full_data_block1.pickle", 'rb') as handle:
            refs_block1, _, _, _, emgs_block1, _, _, _, _, _, _ = pickle.load(handle)
    else:
        raise("Dataset not supported")
            
    if is_train:
        my_user = all_user_keys[idx]
        upper_bound = round(test_split*(emgs_block1[my_user][condition_number,:,:].shape[0]))
        return CustomEMGDataset(emgs_block1[my_user][condition_number,:upper_bound,:], refs_block1[my_user][condition_number,:upper_bound,:])
    
    else:
        my_user = all_user_keys[idx]
        upper_bound = round(test_split*(emgs_block1[my_user][condition_number,:,:].shape[0]))
        return CustomEMGDataset(emgs_block1[my_user][condition_number,upper_bound:,:], refs_block1[my_user][condition_number,upper_bound:,:])


def read_client_data(dataset, idx, is_train=True, condition_number=1, test_split=0.2, test_split_each_update=False):
    # KAI'S EDITED VERSION WHICH IS NOW USED IN THE CODE
    
    dataset_obj = read_data(dataset, idx, is_train, condition_number=condition_number, test_split=test_split, test_split_each_update=test_split_each_update)
    X_data = torch.Tensor(dataset_obj['x']).type(torch.float32)
    y_data = torch.Tensor(dataset_obj['y']).type(torch.float32)
    zipped_data = [(x, y) for x, y in zip(X_data, y_data)]
    return zipped_data
    
    
######################################################################################

def read_data_archive(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data_archive(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
    