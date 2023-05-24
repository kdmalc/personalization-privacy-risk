import torch
import pandas as pd
import numpy as np

import pickle

from utils.emg_dataset_class import CustomEMGDataset

def make_users(condition_number=1, dataset='CPHS', all_user_keys=['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']):
    if dataset.upper()=='CPHS':
        with open(r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\Data\continuous_full_data_block1.pickle", 'rb') as handle:
            refs_block1, _, _, _, emgs_block1, _, _, _, _, _, _ = pickle.load(handle)
    else:
        print("Not supported yet")
    
    #if (type(condition_number) is not int) and len(condition_number)>1:
    #    pass
    if type(condition_number) is int:
        condition_number = [condition_number]
        
    num_users = len(all_user_keys)
    num_conds = len(condition_number)
    print(f"Creating {num_users*num_conds} user nodes")
    user_list = [0]*num_conds*num_users
    
    user_count_idx = -1  # Start at -1 so that first index can be 0
    for my_cond in condition_number:
        for my_user in all_user_keys:
            user_count_idx += 1
            user_list[user_count_idx] = CustomEMGDataset(emgs_block1[my_user][my_cond,:,:], refs_block1[my_user][my_cond,:,:])
            
    return user_list
