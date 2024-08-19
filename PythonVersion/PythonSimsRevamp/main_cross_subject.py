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
from shared_globals import *


# GLOBALS
GLOBAL_METHOD = "PFAFO_GDLS"  #FedAvg #PFAFO_GDLS #NOFL
OPT_METHOD = 'GDLS'  #FULLSCIPYMIN #MaxiterScipyMin #GD #GDLS --> USE GDLS For FedAvg!
# ^ This gets ignored completely when using PFA
NUM_STEPS=3  # This is basically just local_epochs. Num_grad_steps
SCENARIO="CROSS"  # "INTRA" --> Cant be used in this file??

GLOBAL_ROUNDS = 100
BETA=0.01  # Not used with GDLS? Only pertains to PFA regardless
LR=1  # Not used with GDLS?
MAX_ITER=1  # For scipy. Set to -1 for full, otherwise stay with 1
# ^ Do I need to pass this in? Is that not controlled by OPT_METHOD? ...
LOCAL_ROUND_THRESHOLD=75

with open(path+cond0_filename, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)
#with open(path+all_decs_init_filename, 'rb') as fp:
#    init_decoders = pickle.load(fp)    
#cond0_init_decs = [dec[0, :, :] for dec in init_decoders]

# THIS K FOLD SCHEME IS ONLY FOR CROSS-SUBJECT ANALYSIS!!!
# Define number of folds
k = NUM_KFOLDS
kf = KFold(n_splits=k)
# Assuming cond0_training_and_labels_lst is a list of labels for 14 clients
user_ids = list(range(14))
folds = list(kf.split(user_ids))
#cross_val_res_lst = [[0, 0]]*NUM_KFOLDS
# ^ THIS IS BAD CODE! 
## creates a list of references to the same inner list. 
## This means that when you modify one element, all elements change.
## Instead, use list comprehension:
cross_val_res_lst = [[0, 0] for _ in range(NUM_KFOLDS)]

for fold_idx, (train_ids, test_ids) in enumerate(folds):
    print(f"Fold {fold_idx+1}/{k}")
    print(f"{len(train_ids)} Train_IDs: {train_ids}")
    print(f"{len(test_ids)} Test_IDs: {test_ids}")
    
    # Initialize clients for training
    train_clients = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM,
                            beta=BETA, scenario=SCENARIO, local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, current_fold=fold_idx, num_kfolds=NUM_KFOLDS, global_method=GLOBAL_METHOD, max_iter=MAX_ITER, 
                            num_steps=NUM_STEPS, use_zvel=USE_HITBOUNDS, test_split_type=TEST_SPLIT_TYPE) for i in train_ids]
    # Initialize clients for testing
    test_clients = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM,
                           beta=BETA, scenario=SCENARIO, local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, current_fold=fold_idx, availability=False, val_set=True, num_kfolds=NUM_KFOLDS, global_method=GLOBAL_METHOD, max_iter=MAX_ITER, 
                           num_steps=NUM_STEPS, use_zvel=USE_HITBOUNDS, test_split_type=TEST_SPLIT_TYPE) for i in test_ids]

    testing_datasets_lst = []
    for test_cli in test_clients:
        testing_datasets_lst.append(test_cli.get_testing_dataset())
    for train_cli in train_clients:
        train_cli.set_testset(testing_datasets_lst)

    full_client_lst = train_clients+test_clients

    server_obj = Server(1, copy.deepcopy(D_0), opt_method=OPT_METHOD, global_method=GLOBAL_METHOD, all_clients=full_client_lst)
    server_obj.current_fold = fold_idx
    server_obj.global_rounds = GLOBAL_ROUNDS
    for i in range(GLOBAL_ROUNDS):
        if i % 10 == 0:
            print(f"Round {i} of {GLOBAL_ROUNDS}")

        server_obj.execute_FL_loop()

        #if i % 10 == 0:
        #    print(f"Global test loss: {server_obj.global_test_error_log[-1]}")
        #    print(f"Local test loss: {server_obj.local_test_error_log[-1]}")

    if PLOT_EACH_FOLD:
        plt.figure()  # Create a new figure
        plt.plot(server_obj.local_train_error_log, alpha=0.5, label=f"f{fold_idx} local train")
        plt.plot(server_obj.local_test_error_log, alpha=0.5, label=f"f{fold_idx} local test")
        if server_obj.global_method!="NOFL":
            plt.plot(server_obj.global_train_error_log, alpha=0.5, label=f"f{fold_idx} global train")
            plt.plot(server_obj.global_test_error_log, alpha=0.5, label=f"f{fold_idx} global test")
        plt.xlabel("Training Round")
        plt.ylabel("Loss")
        plt.title(f"Train/Test Local/Global Error Curves For Fold {fold_idx}")
        plt.legend()
        plt.savefig(server_obj.trial_result_path + f'\\TrainTestLossCurvesf{fold_idx}.png', format='png')
        plt.show()

    # Record results
    # copy.deepcopy didn't fix it... must be upstream...
    cross_val_res_lst[fold_idx][0] = copy.deepcopy(server_obj.local_train_error_log)
    cross_val_res_lst[fold_idx][1] = copy.deepcopy(server_obj.local_test_error_log)

    #server_obj.save_results_h5(save_cost_func_comps=False, save_gradient=False)
    # Save the model for the current fold
    if GLOBAL_METHOD.upper()!="NOFL":
        print("Saving server's final (global) model")
        # TODO: Where is model_saving_dir defined...................
        dir_path = os.path.join(model_saving_dir, server_obj.str_current_datetime, GLOBAL_METHOD)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, f'servers_final_model_fold{fold_idx}.npy'), server_obj.w)

# Plot all results:
plt.figure()  # Create a new figure
running_train_loss = np.zeros(GLOBAL_ROUNDS)
running_test_loss = np.zeros(GLOBAL_ROUNDS)
for fold_idx in range(NUM_KFOLDS):
    train_loss = cross_val_res_lst[fold_idx][0]
    test_loss = cross_val_res_lst[fold_idx][1]

    plt.plot(train_loss, alpha=0.5, label=f"Fold{fold_idx} Train")
    plt.plot(test_loss, alpha=0.5, label=f"Fold{fold_idx} Test")

    running_train_loss += np.array(train_loss)
    running_test_loss += np.array(test_loss)
# Average to get cross val curve:
avg_cv_train_loss = running_train_loss / NUM_KFOLDS
avg_cv_test_loss = running_test_loss / NUM_KFOLDS
plt.plot(avg_cv_train_loss, linewidth=2, label="Avg CrossVal Train")
plt.plot(avg_cv_test_loss, linewidth=2, label="Avg CrossVal Test")
plt.xlabel("Training Round")
plt.ylabel("Loss")
plt.title("Train/Test Local Error Curves")
plt.legend()
plt.savefig(server_obj.trial_result_path + '\\TrainTestLossCurves.png', format='png')
plt.show()
    
server_obj.save_header()
with h5py.File(server_obj.h5_file_path + "_CrossValResults.h5", 'w') as hf:
    for fold_idx in range(NUM_KFOLDS):
        #if server_obj.global_method!="NOFL":
        #    hf.create_dataset('global_test_error_log', data=self.global_test_error_log)
        #    hf.create_dataset('global_train_error_log', data=self.global_train_error_log)
        hf.create_dataset(f'Fold{fold_idx}_local_test_error_log', data=cross_val_res_lst[fold_idx][1])
        hf.create_dataset(f'Fold{fold_idx}_local_train_error_log', data=cross_val_res_lst[fold_idx][0])
    # Save the averaged cv results
    hf.create_dataset(f'AveragedCV_local_test_error_log', data=avg_cv_test_loss)
    hf.create_dataset(f'AveragedCV_local_train_error_log', data=avg_cv_train_loss)

