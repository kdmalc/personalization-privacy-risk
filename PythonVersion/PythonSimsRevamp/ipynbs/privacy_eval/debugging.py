import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle
import random
import os
import h5py
import copy
import statistics

import seaborn as sns

final_update = 16

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import re
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score

#import warnings
#warnings.filterwarnings("ignore")

random.seed(a=1)

def load_final_model_performances(cv_results_path, filename, num_clients=14, num_folds=7, verbose=False):
    extraction_dict = dict()
    for i in range(num_folds):
        h5_path = os.path.join(cv_results_path, filename+f"{i}.h5")
        #print(h5_path)
        
        # Load data from HDF5 file
        with h5py.File(h5_path, 'r') as f:
            a_group_key = list(f.keys())
            #if i==0:
            #    print(a_group_key)
            for key in a_group_key:
                #print(key)
        
                if key=="client_local_model_log":
                    client_keys = list(f[key])
                    #print(client_keys)
                    for ck in client_keys:
                        ed_key = f"{ck}_fold{i}"  # Does this never update from or something...
                        #print(f"Key: {key}, Client: {ck}, Fold: {i}")
    
                        # So this doenst have any knoledge of the fold number???
                        if len(list(f[key][ck]))==0:
                            #print(f"{ed_key} SKIPPED!")
                            pass
                        else:
                            #print(f"{ed_key} SUCCESS!")
                            extraction_dict[ed_key] = list(f[key][ck])
                elif key=="global_model_log" and "NOFL" not in filename:
                    # Do I need to turn this off for NoFL? Or will it just be empty and append something empty...
                    ed_key = f"{key}_fold{i}"
                    #print(ed_key)
                    extraction_dict[ed_key] = list(f[key])
                else:
                    pass

    return extraction_dict


NUM_PLOTTED_CONDITIONS = 6
NUM_PLOTTED_CONDITIONS_WITH_GLOBAL = 10
NUM_CLIENTS = 14
NUM_FOLDS = 5

results_path = r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results'
current_directory = r'\FinalSims'
base_path = results_path + current_directory

# CROSS
cpfa_model_dict = load_final_model_performances(base_path+r'\09-19_23-07_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_KFold')
cfa_model_dict = load_final_model_performances(base_path+r'\09-19_23-25_FEDAVG', 'GDLS_FEDAVG_KFold')
cnofl_model_dict = load_final_model_performances(base_path+r'\09-19_23-30_NOFL', 'FULLSCIPYMIN_NOFL_KFold')
# INTRA
ipfa_model_dict = load_final_model_performances(base_path+r'\09-19_23-05_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_KFold')
ifa_model_dict = load_final_model_performances(base_path+r'\09-19_23-04_FEDAVG', 'GDLS_FEDAVG_KFold')
inofl_model_dict = load_final_model_performances(base_path+r'\09-19_22-56_NOFL', 'FULLSCIPYMIN_NOFL_KFold')


def execute_linkage_attack(extractration_dict, num_kfolds=7, stratification=False, i_modulus=3, i_lt_or_eq='eq', 
                           my_metrics_columns=['Algorithm', 'Update Number', 'CV Acc', 'Test Acc'], 
                           my_models_list=[KNeighborsClassifier(), DecisionTreeClassifier(), GradientBoostingClassifier()], 
                           make_df=True, full_test_df=pd.DataFrame(), min_n_samples=10):

    keys = extractration_dict.keys()
    num_updates_lst = []
    for key in keys:
        num_updates_lst.append(len(extractration_dict[key]))
        # ORIGINAL CODE:
        #num_updates_lst.append(len(extractration_dict[list(extractration_dict.keys())[0]]))
    mode_update = statistics.mode(num_updates_lst)
    max_update = max(num_updates_lst)
    if max_update == mode_update:  # Eg this is a poor man's proxy for if this is the NOFL case
        num_updates = mode_update
    else:
        num_updates = min(max_update, int(statistics.mean(num_updates_lst) + statistics.stdev(num_updates_lst)))  # int() rounds down
    print(f"{num_updates} unique models")
    
    # Initialize a list of empty DataFrames for each user group
    dec_flattened_list = [pd.DataFrame(columns=['Subject', 'Fold', 'Update Number', 'Flattened Dec']) for _ in range(num_updates)]
    
    # Regular expression pattern to extract subject and fold
    #pattern = r"S(\d+)_client_local_model_log_fold(\d+)"
    pattern = r"(S\d+)_client_local_model_log_fold(\d+)"
    
    # Loop through keys and updates to populate the DataFrames
    for key_idx, key in enumerate(keys):
        key_len = len(extractration_dict[key])
        match = re.search(pattern, key)  # Extract the subject and fold using regex
        if match:
            #subject = int(match.group(1))  # e.g., '0', '1', '10'
            subject = match.group(1)  # e.g., 'S0', 'S1', 'S10'
            fold = int(match.group(2))  # e.g., '0', '1', '2'
            for update_number in range(num_updates): 
                if update_number >= key_len:
                    continue
                else:
                    user_data = np.ravel(extractration_dict[key][update_number])
                    dec_flattened_list[update_number].loc[len(dec_flattened_list[update_number])] = [subject, fold, update_number, user_data]
    
    # Concat all the dfs into a single training input dataframe
    dec_flattened = pd.concat(dec_flattened_list, ignore_index=True, axis=0)
    flattened_input_df = dec_flattened.join(dec_flattened['Flattened Dec'].apply(pd.Series)).drop('Flattened Dec', axis=1)
    #print(expanded_dec_df.shape)
    #expanded_dec_df.head()
    
    full_test_res_df = pd.DataFrame(columns=my_metrics_columns)
    
    # Adjust stop index to prevent overflow
    n_start_stop=(0, num_updates)
    n_stop = n_start_stop[1] - 1
    print(f"num_updates {num_updates}; n_start_stop {n_start_stop}; new n_stop {n_stop}")
    
    for i in range(n_start_stop[0], n_stop + 1):
        #print(f"Round i={i} of stop={n_start_stop[1]}")
    
        if make_df:
            #print("Making custom test set, NOT USING PASSED IN full_test_df")
    
            if i%i_modulus==0:
                print(f"Round i={i} of stop={n_start_stop[1]}")
                
                # Filter the dataframe to use data from updates <= i
                # TODO: DECIDE ON USING <, <=, OR BOTH!
                if i_lt_or_eq == 'lt':
                    train_df = flattened_input_df[flattened_input_df['Update Number'] <= i]
                elif i_lt_or_eq == 'eq':
                    train_df = flattened_input_df[flattened_input_df['Update Number'] == i]
                test_df = flattened_input_df[flattened_input_df['Update Number'] == n_stop]  # Hold out the last update for testing
            else:
                # SKIP THE ENTIRE LOOP
                #print(f"{i} skipped for speed!")
                continue
        else:
            raise ValueError('Need to set train_df somehow...: for now, only use make_df=True')
        
        X_test = test_df.drop(columns=['Subject', 'Fold', 'Update Number'])
        y_test = test_df['Subject']
    
        # Explicitly use the 'Fold' column for cross-validation
        for model in my_models_list:
            cv_scores = []
            for fold_num in range(num_kfolds):
                #print(f"Fold {fold_num}")
                train_fold = train_df[train_df['Fold'] != fold_num]
                val_fold = train_df[train_df['Fold'] == fold_num]
    
                if train_fold.shape[0] < min_n_samples:
                    # if model is KNN and num_train_samples < model.n_neighbors
                    # Adjust n_neighbors if necessary
                    #print(f"Adjusting n_neighbors to {num_train_samples} since it's smaller than n_neighbors.")
                    #model.set_params(n_neighbors=num_train_samples) 
                    
                    print(f"Not enough samples! Skipping this ruond")
                    continue
                elif val_fold.shape[0]==0:
                    # Clients are trained a different number of rounds, so the max may be much higher than the average
                    # Thus resulting in upper rounds not containing any instances of a specific fold (eg can't train the model the way it is expected here)
                    continue

                X_train_fold = train_fold.drop(columns=['Subject', 'Fold', 'Update Number']).reset_index(drop=True)
                y_train_fold = train_fold['Subject'].reset_index(drop=True)
    
                X_val_fold = val_fold.drop(columns=['Subject', 'Fold', 'Update Number'])
                y_val_fold = val_fold['Subject']
    
                # Fit model on the current training fold
                model.fit(X_train_fold, y_train_fold)
    
                # Evaluate on the validation fold
                score = model.score(X_val_fold, y_val_fold)
                cv_scores.append(score)
    
            # After evaluating all folds, calculate the average cross-validation score
            cv_avg_score = sum(cv_scores) / len(cv_scores)
    
            # Calculate the test accuracy on the hold-out test set
            test_acc = model.score(X_test, y_test)
    
            # Save the results to the full_test_res_df DataFrame
            new_row = pd.DataFrame({
                'Algorithm': [type(model).__name__],
                'Update Number': [i],
                'CV Acc': [cv_avg_score],
                'Test Acc': [test_acc]
            })
            full_test_res_df = pd.concat([full_test_res_df, new_row], ignore_index=True)
    
    return full_test_res_df

i_nofl = 5
i_fl = 50

cpfa_res_df = execute_linkage_attack(cpfa_model_dict, i_modulus=i_fl)
#cfa_res_df = execute_linkage_attack(cfa_model_dict, i_modulus=i_fl)
#cnofl_res_df = execute_linkage_attack(cnofl_model_dict, i_modulus=i_nofl)
#ipfa_res_df = execute_linkage_attack(ipfa_model_dict, i_modulus=i_fl)
#ifa_res_df = execute_linkage_attack(ifa_model_dict, i_modulus=i_fl)
#inofl_res_df = execute_linkage_attack(inofl_model_dict, i_modulus=i_nofl)