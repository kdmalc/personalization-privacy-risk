from itertools import combinations
import numpy as np
import os
import numpy as np
import random
import pickle
import copy
from datetime import datetime
import h5py

np.random.seed(0)
random.seed(0)

from experiment_params import *
from cost_funcs import *
from fl_sim_client import *
from fl_sim_server import *
from shared_globals import *

# THIS CODE WOULD REPLACE CROSS USER MAIN TO GENERATE ALL THE MEMBERSHIP INFERENCE MODELS
## Not doing K Folds, don't care here

# GLOBALS
PRIVACY_ATTACK = True
GLOBAL_METHOD = "FedAvg"  #FedAvg #PFAFO_GDLS #NOFL
OPT_METHOD = 'FULLSCIPYMIN' if GLOBAL_METHOD=="NOFL" else 'GDLS'
GLOBAL_ROUNDS = 12 if GLOBAL_METHOD=="NOFL" else 250
LOCAL_ROUND_THRESHOLD = 1 if GLOBAL_METHOD=="NOFL" else 20
NUM_STEPS = 3  # This is basically just local_epochs. Num_grad_steps
SCENARIO = "CROSS"  # "INTRA" cant be used in this file!

BETA=0.01  # Not used with GDLS? Only pertains to PFA regardless
LR=1  # Not used with GDLS?

# OVERWRITES
## Actually, leave these at their default values. In cross it doesnt matter since kfolds are handled in the main func (eg I can just not run them here)
#USE_KFOLDCV = True
#TEST_SPLIT_TYPE = 'KFOLDCV'

with open(path+cond0_filename, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)

# Create combinations of users to hold out in each fold (2 users at a time)
user_pairs = list(combinations(range(NUM_USERS), 2))
#user_pairs = user_pairs[:4]
num_user_pairs = len(user_pairs)

# Initialize storage for results
membership_inference_results = {}

# Loop over each user pair combination to leave them out
for pair_idx, (user1, user2) in enumerate(user_pairs):
    print(f"PAIR_IDX: {pair_idx}/{num_user_pairs}\nTraining models leaving out users {user1} and {user2}")
    
    # Get the list of users excluding the pair
    remaining_users = [user for user in range(NUM_USERS) if user != user1 and user != user2]
    
    # Train a model on the remaining users
    train_clients = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM,
                            beta=BETA, scenario=SCENARIO, local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, 
                            num_kfolds=len(user_pairs), global_method=GLOBAL_METHOD, max_iter=MAX_ITER, num_steps=NUM_STEPS) 
                            for i in remaining_users]
    
    # Initialize clients for testing (the two held-out users)
    test_clients = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM,
                            beta=BETA, scenario=SCENARIO, local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, 
                            availability=False, val_set=True, num_kfolds=len(user_pairs), global_method=GLOBAL_METHOD, max_iter=MAX_ITER, num_steps=NUM_STEPS) 
                            for i in [user1, user2]]

    # Store testing datasets from held-out clients
    testing_datasets_lst = [test_cli.get_testing_dataset() for test_cli in test_clients]
    
    # Pass the testing datasets to the train clients
    for train_cli in train_clients:
        train_cli.set_testset(testing_datasets_lst)
    
    # Combine all clients (train + test) for federated learning
    full_client_lst = train_clients + test_clients

    # Initialize server
    server_obj = Server(1, copy.deepcopy(D_0), opt_method=OPT_METHOD, global_method=GLOBAL_METHOD, privacy_attack=PRIVACY_ATTACK, all_clients=full_client_lst)
    server_obj.set_save_filename(CURRENT_DATETIME)
    server_obj.global_rounds = GLOBAL_ROUNDS

    # Run federated learning rounds
    for i in range(GLOBAL_ROUNDS):
        server_obj.execute_FL_loop()

    # Save results after the training
    server_obj.save_results_h5(dir_str=f"model_without_users_{user1}_and_{user2}")

    # Set the held out users / testing clients training datasets
    reset_heldout_clients = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM,
                        beta=BETA, scenario=SCENARIO, local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, 
                        num_kfolds=len(user_pairs), global_method=GLOBAL_METHOD, max_iter=MAX_ITER, num_steps=NUM_STEPS) 
                        for i in [user1, user2]]
    all_cli_objs_lst = train_clients + reset_heldout_clients
    
    # Perform membership inference on the held-out users
    # Test each global model on every client's training data
    performance_metrics = [0 for _ in range(NUM_USERS)]
    #for client_id in range(NUM_USERS):
        # Store the loss in the performance_metrics dictionary
        #performance_metrics[f"fold_{pair_idx}"] = performance_metrics.get(f"fold_{pair_idx}", [])
        #performance_metrics[f"fold_{pair_idx}"].append(loss)
    # Evaluate the model on the client's training dataset
    for cli in all_cli_objs_lst:
        #loss1 = server_obj.evaluate(reset_heldout_clients[0])
        #loss2 = server_obj.evaluate(reset_heldout_clients[1])
        performance_metrics[cli.ID] = server_obj.evaluate_client(cli)

    # After evaluating all clients, you can store or process performance_metrics further as needed
    membership_inference_results[f"model_without_users_{user1}_and_{user2}"] = performance_metrics

# Output the results for analysis
#print("Membership Inference Results:", membership_inference_results)

# Save the membership_inference_results to a pickle file
pickle_filename = server_obj.trial_result_path + "\\membership_inference_results.pkl"
with open(pickle_filename, "wb") as f:
    pickle.dump(membership_inference_results, f)
print(f"Results saved to {pickle_filename}")
