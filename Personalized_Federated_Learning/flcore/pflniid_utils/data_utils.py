# PFLNIID

import numpy as np
import os
import torch
import pickle
from utils.emg_dataset_class import *

# This is depreciated
def read_data(dataset, client_idx, update_number, is_train=True, condition_idx=0, test_split=0.2, test_split_each_update=False):
    assert("Tried to run read_data()")
    # KAI'S EDITED VERSION WHICH IS NOW USED IN THE CODE
    
    # Some data
    update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
    all_user_keys = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']

    # Print some stuff
    print(f"Client{client_idx}, train={is_train}: read_data() called!")
    if test_split_each_update:
        raise("Not supported yet.  Idk if this is necessary :/")
    
    # Load our data
    if dataset.upper()=='CPHS':
        with open(r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\Data\continuous_full_data_block1.pickle", 'rb') as handle:
            refs_block1, _, _, _, emgs_block1, _, _, _, _, _, _ = pickle.load(handle)
    else:
        raise("Dataset not supported")
        
    my_user = all_user_keys[client_idx]
    upper_bound = round((1-test_split)*(emgs_block1[my_user][condition_idx,:,:].shape[0]))
    # The below gets stuck in the debugger and just keeps running until you step over
    train_test_update_number_split = min(update_ix, key=lambda x:abs(x-upper_bound))
    max_training_update = update_ix.index(train_test_update_number_split)
    max_training_update_lb = max_training_update-1
    
    # PRE UPDATE INCLUSION
    #if is_train:
    #    return CustomEMGDataset(emgs_block1[my_user][condition_idx,:upper_bound,:], refs_block1[my_user][condition_idx,:upper_bound,:])
    #else:
    #    return CustomEMGDataset(emgs_block1[my_user][condition_idx,upper_bound:,:], refs_block1[my_user][condition_idx,upper_bound:,:])
    
    if update_number < max_training_update:
        update_lower_bound = update_ix[update_number]
        update_upper_bound = update_ix[update_number+1]
    else:
        update_lower_bound = max_training_update_lb
        update_upper_bound = max_training_update
    #print(f"Printing update upper bound for some reason: {update_upper_bound}")
    
    if is_train:
        return CustomEMGDataset(emgs_block1[my_user][condition_idx,update_lower_bound:update_upper_bound,:], refs_block1[my_user][condition_idx,update_lower_bound:update_upper_bound,:])
    else:
        return CustomEMGDataset(emgs_block1[my_user][condition_idx,update_upper_bound:,:], refs_block1[my_user][condition_idx,update_upper_bound:,:])
    

def read_client_data(dataset, client_idx, update_number, is_train=True, condition_idx=1, test_split=0.2, test_split_each_update=False):
    assert("Tried to run read_client_data()")
    # KAI'S EDITED VERSION WHICH IS NOW USED IN THE CODE
    print(f"Client{client_idx}, train={is_train}: read_CLIENT_data() called!")
    
    dataset_obj = read_data(dataset, client_idx, update_number, is_train=is_train, condition_idx=condition_idx, test_split=test_split, test_split_each_update=test_split_each_update)
    X_data = torch.Tensor(dataset_obj['x']).type(torch.float32)
    # The below line (or this line??) gets stuck in the debugger too. Have to step out and get to the space above idk
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
    