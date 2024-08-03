import os
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import time
import pickle
from sklearn.model_selection import KFold
import copy

from matplotlib import pyplot as plt
#import seaborn as sns
#from presentation_sns_config import *

from experiment_params import *
from cost_funcs import *
from fl_sim_client import *
from fl_sim_server import *

# GLOBALS
path = r'C:\Users\kdmen\Desktop\Research\Data\CPHS_EMG'
cond0_filename = r'\cond0_dict_list.p'
all_decs_init_filename = r'\all_decs_init.p'
nofl_decs_filename = r'\nofl_decs.p'
id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
implemented_client_training_methods = ['GD', 'FullScipyMin', 'MaxIterScipyMin']
num_participants = 14
USE_KFOLDCV = True
SCENARIO = 'CROSS-SUBJECT' # "INTRA-SUBJECT"
GLOBAL_METHOD = "FEDAVG"
# For exclusion when plotting later on
bad_nodes = [] #[1,3,13]
D_0 = np.random.rand(2,64)
num_updates = 18
step_indices = list(range(num_updates))

with open(path+cond0_filename, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)
#with open(path+all_decs_init_filename, 'rb') as fp:
#    init_decoders = pickle.load(fp)    
#cond0_init_decs = [dec[0, :, :] for dec in init_decoders]

# THIS K FOLD SCHEME IS ONLY FOR CROSS-SUBJECT ANALYSIS!!!

# Define number of folds
k = 5
kf = KFold(n_splits=k)

# Assuming cond0_training_and_labels_lst is a list of labels for 14 clients
user_ids = list(range(14))
folds = list(kf.split(user_ids))

for fold_idx, (train_ids, test_ids) in enumerate(folds):
    print(f"Fold {fold_idx+1}/{k}")
    print(f"{len(train_ids)} Train_IDs: {train_ids}")
    print(f"{len(test_ids)} Test_IDs: {test_ids}")
    
    # Initialize clients for training
    train_clients = [Client(i, copy.deepcopy(D_0), 'MaxiterScipyMin', cond0_training_and_labels_lst[i], 'streaming', starting_update=10, use_kfoldv=True, global_method='FedAvg', max_iter=1, num_steps=1, use_zvel=False, test_split_type='end', use_up16_for_test=False) for i in train_ids]

    # Initialize clients for testing
    test_clients = [Client(i, copy.deepcopy(D_0), 'MaxiterScipyMin', cond0_training_and_labels_lst[i], 'streaming', starting_update=10, availability=False, use_kfoldv=True, global_method='FedAvg', max_iter=1, num_steps=1, use_zvel=False, test_split_type='end', use_up16_for_test=False) for i in test_ids]

    testing_datasets_lst = []
    for test_cli in test_clients:
        testing_datasets_lst.append(test_cli.get_testing_dataset())
    for train_cli in train_clients:
        train_cli.set_testset(testing_datasets_lst)

    full_client_lst = train_clients+test_clients

    global_model_1scipystep = Server(1, copy.deepcopy(D_0), opt_method='FullScipyMin', global_method='FedAvg', all_clients=full_client_lst)
    
    big_loop_iters = 50
    for i in range(big_loop_iters):
        if i % 10 == 0:
            print(f"Round {i} of {big_loop_iters}")
            # Ought to print the loss too...
        global_model_1scipystep.execute_FL_loop()
    

