import os
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import time
import pickle
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
USE_KFOLDCV = True
GLOBAL_METHOD = "NOFL"
OPT_METHOD = 'FULLSCIPYMIN'
GLOBAL_ROUNDS = 50
LR=0.1
MAX_ITER=None  # Setting to -1 DOES NOT WORK FOR THIS CODE BASE! Use OPT_METHOD to specify that instead...
NUM_STEPS=1  # This is also basically just local_epochs, since I don't batch. Num_grad_steps
TEST_SPLIT_TYPE='kfoldcv'

with open(path+cond0_filename, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)
#with open(path+all_decs_init_filename, 'rb') as fp:
#    init_decoders = pickle.load(fp)    
#cond0_init_decs = [dec[0, :, :] for dec in init_decoders]

cross_val_res_lst = [[0, 0] for _ in range(NUM_KFOLDS)]
for fold_idx in range(NUM_KFOLDS):
    print(f"Fold {fold_idx+1}/{NUM_KFOLDS}")
    
    # Initialize clients for training
    full_client_lst = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i],
                            DATA_STREAM, current_fold=fold_idx, global_method=GLOBAL_METHOD, max_iter=MAX_ITER, 
                            num_steps=NUM_STEPS, use_zvel=USE_HITBOUNDS, test_split_type=TEST_SPLIT_TYPE) for i in range(NUM_USERS)]

    server_obj = Server(1, copy.deepcopy(D_0), opt_method=OPT_METHOD, global_method=GLOBAL_METHOD, all_clients=full_client_lst)
    # Add these to the init func params...
    server_obj.current_fold = fold_idx
    server_obj.global_rounds = GLOBAL_ROUNDS
    for i in range(GLOBAL_ROUNDS):
        #if i % 10 == 0:
        #    print(f"Round {i} of {GLOBAL_ROUNDS}")

        server_obj.execute_FL_loop()

        #if i % 10 == 0:
        #    # NOTE: There are no global error logs for NOFL/LOCAL!
        #    print(f"Local test loss: {server_obj.local_test_error_log[-1]}")

    if PLOT_EACH_FOLD:
        plt.figure()  # Create a new figure
        plt.plot(server_obj.local_train_error_log, label=f"f{fold_idx} local train")
        plt.plot(server_obj.local_test_error_log, label=f"f{fold_idx} local test")
        plt.xlabel("Training Round")
        plt.ylabel("Loss")
        plt.title(f"Train/Test Local Error Curves For Fold {fold_idx}")
        plt.legend()
        plt.savefig(server_obj.trial_result_path + f'\\TrainTestLossCurvesf{fold_idx}.png', format='png')
        plt.show()

    # Record results
    # copy.deepcopy didn't fix it... must be upstream...
    cross_val_res_lst[fold_idx][0] = copy.deepcopy(server_obj.local_train_error_log)
    cross_val_res_lst[fold_idx][1] = copy.deepcopy(server_obj.local_test_error_log)

    #server_obj.save_results_h5(save_cost_func_comps=False, save_gradient=False)

# Plot all results:
plt.figure()  # Create a new figure
running_train_loss = np.zeros(GLOBAL_ROUNDS)
running_test_loss = np.zeros(GLOBAL_ROUNDS)
for fold_idx in range(NUM_KFOLDS):
    train_loss = cross_val_res_lst[fold_idx][0]
    test_loss = cross_val_res_lst[fold_idx][1]

    plt.plot(train_loss, label=f"Fold{fold_idx} Train")
    plt.plot(test_loss, label=f"Fold{fold_idx} Test")

    running_train_loss += np.array(train_loss)
    running_test_loss += np.array(test_loss)
# Average to get cross val curve:
avg_cv_train_loss = running_train_loss / NUM_KFOLDS
avg_cv_test_loss = running_test_loss / NUM_KFOLDS
plt.plot(avg_cv_train_loss,  label="Avg CrossVal Train")
plt.plot(avg_cv_test_loss,  label="Avg CrossVal Test")
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
