#!/usr/bin/env python
#import copy

# Set a seed for reproducibility!
import torch
torch.manual_seed(0)
# ^seed the RNG for all devices (both CPU and CUDA)
#torch.use_deterministic_algorithms(True)
# ^https://pytorch.org/docs/stable/notes/randomness.html

import numpy as np
np.random.seed(0)
import random
random.seed(0)

import copy
import argparse
import os
import time
import warnings
import logging
from utils.helper_funcs import convert_cmd_line_str_lst_to_type_lst

from flcore.pflniid_utils.result_utils import average_data
from flcore.pflniid_utils.mem_utils import MemReporter

from sklearn.model_selection import KFold
from utils.server_init_models_algos import *
from utils.emg_dataset_class import *


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def create_user_folds(users, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    user_folds = list(kf.split(users))
    return user_folds


def run_kfcv(args):
    # Need to figure out file saving since it is running 5 times... does it overwrite? 
    # Maybe times is for statistics/repeatability and not just PFL... double check what it is doing and how they used it...

    time_list = []
    reporter = MemReporter()

    user_IDs = args.all_subj_IDs
    # This needs double checked...
    user_folds = create_user_folds(user_IDs, args.num_kfold_splits)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(user_folds):
        print(f"Fold {fold + 1}/{args.num_kfold_splits}")

        #if fold!=0:
        #    # For the first run (0), server is already set above
        #    server = init_algo(args)
        args.model = init_model(args)
        server = init_algo(args)
        server.current_fold += 1
        if fold > server.num_max_kfold_splits:
            print(f"Max kfold ({server.num_max_kfold_splits}) has been achieved, skipping the rest of the folds for speed")
            continue
        
        # Set training and validation users
        train_user_IDs = [user_IDs[i] for i in train_idx]
        val_user_IDs = [user_IDs[i] for i in val_idx]
        server.train_subj_IDs = train_user_IDs
        server.test_subj_IDs = val_user_IDs

        val_dataset_lst = []
        for val_cli_subjID in val_user_IDs:
            val_cli = server.dict_map_subjID_to_clientobj[val_cli_subjID]
            val_dataset_lst.append(val_cli.load_test_data())
        
        testloader = create_unified_fold_test_dataloader(val_dataset_lst, server.batch_size)
        assert(len(testloader)!=0)
        for cli_ID in server.all_subj_IDs:
            cli_obj = server.dict_map_subjID_to_clientobj[cli_ID]
            cli_obj.testloader = copy.deepcopy(testloader)

        # args.times=1 for now... I'm not using this loop at all actually so I removed it...
        #for i in range(args.prev, args.times):
        #    print(f"\n============= Running time: {i}th =============")
        print(f"\n============= STARTING NEW TRIAL =============")
        start = time.time()
        print(args.model)

        server.train()
        time_list.append(time.time()-start)

        #server.plot_results()
        print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    mean_cv_loss = np.mean(cv_results)
    std_cv_loss = np.std(cv_results)
    print(f"Cross-validation results: {mean_cv_loss:.4f} (+/- {std_cv_loss:.4f})")
    server.plot_results(plot_this_vec=mean_cv_loss, my_title="Mean K-Fold Cross Val Train And Test Loss")
      
    # Global average
    ## Not sure if it should be running this at all, I think the average is already done above?
    #if args.algorithm != "Local":
    #    average_data(server.trial_result_path, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")
    reporter.report()
    return server, cv_results, mean_cv_loss, std_cv_loss


def run(args):
    # Old non k fold version. Not sure if it is compatible with the current code...
    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        ####################################################################################################
        # Generate args.model
        if args.model_str == "LinearRegression":
            args.model = torch.nn.Linear(args.input_size, args.output_size, args.linear_model_bias)  #input_size, output_size, bias boolean
        elif args.model_str == "RNN":
            # Initialize the RNN model
            #rnn_model = RNNModel(D, hidden_size, 2)
            args.model = RNNModel(args.input_size, args.hidden_size, args.output_size)
        elif args.model_str == "LSTM":
            # Initialize the LSTM model
            #hidden_size = 64
            #lstm_model = LSTMModel(D, hidden_size, output_size)
            args.model = LSTMModel(args.input_size, args.hidden_size, args.output_size)
        elif args.model_str == "GRU":
            #args.model = GRUModel(args.input_size, args.hidden_size, args.output_size, args.sequence_length)
            args.model = GRUModel(args.input_size, args.hidden_size, args.output_size)
        elif args.model_str == "Transformer":
            args.model = TransformerModel(args.input_size, args.output_size)
        else:
            raise NotImplementedError
        ####################################################################################################

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            # FIX ARGS.HEAD --> LINEAR REGRESSION HAS NO FC
            #args.head = copy.deepcopy(args.model.fc)
            #args.model.fc = nn.Identity()
            #args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == "Local":
            server = Local(args, i)
        elif args.algorithm == "APFL":
            server = APFL(args, i)
        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)
        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)
        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time()-start)

    server.plot_results()
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
      
    # Global average
    if args.algorithm != "Local":
        average_data(server.trial_result_path, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
        # Not super sure what "times" is, by default it is 1. Assuming it runs the process multiple times to average out the stochasticity?

    print("All done!")
    reporter.report()
    return server


def parse_args():
    parser = argparse.ArgumentParser()

    # THINGS I AM CURRENTLY CHANGING A LOT
    #Local #FedAvg #APFL #pFedMe (not working) ## #PerAvg #Centralized
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-bt', "--beta", type=float, default=0.001,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.3,
                        help="Fraction of clients to be active in training per round")
    parser.add_argument('-lrt', "--local_round_threshold", type=int, default=50,
                        help="Number of communication rounds per client until a client will advance to the next batch of streamed data")
    parser.add_argument('-m', "--model_str", type=str, default="LinearRegression")  
    parser.add_argument('-lbs', "--batch_size", type=int, default=600)
    # For non-deep keep 1202: --> Idk if this is necessary actually, I think it will work regardless
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=5)  # KAI: Originally was 2000 --> That's way too much for cross val lol
    parser.add_argument('-stup', "--starting_update", type=int, default=10,
                        help="Which update to start on (for CPHS Simulation). Use 0 or 10.")
    parser.add_argument('-save_mdls', "--save_models", type=bool, default=False) # Uhhh what does this do...
    
    # CONTINUAL LEARNING
    parser.add_argument('-ewc_bool', "--ewc_bool", type=bool, default=False)
    parser.add_argument('-fisher_mult', "--fisher_mult", type=int, default=1e3)
    parser.add_argument('-optimizer_str', "--optimizer_str", type=str, default="SGD")

    # THESE ARE NOT CHANGING FREQUENTLY:
    ##################################################################################################################################

    # DEEP LEARNING STUFF
    # num_layers isn't used right now... need to add a way to make this hidden sizes a list or something...
    parser.add_argument('-num_layers', "--num_layers", type=int, default=1)
    parser.add_argument('-input_size', "--input_size", type=int, default=64)
    parser.add_argument('-hidden_size', "--hidden_size", type=int, default=32)
    parser.add_argument('-sequence_length', "--sequence_length", type=int, default=1)
    parser.add_argument('-output_size', "--output_size", type=int, default=2)

    # Switch this to block 2!
    parser.add_argument('-con_num', "--condition_number_lst", type=str, default='[3]', # Use 3 and/or 7
                        help="Which condition number (trial) to train on. Must be a list. By default, will iterate through all train_subjs for each cond (eg each cond_num gets its own client even for the same subject)")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="How many times a client should iterate through their current update dataset.")  
    parser.add_argument('-ngradsteps', "--num_gradient_steps", type=int, default=1, 
                        help="How many gradient steps in one local epoch.")  
    
    # SECTION: Test Split Related
    # ^ GETTING REPLACED BY K-FOLD CROSS VAL!
    parser.add_argument('-test_split_fraction', "--test_split_fraction", type=float, default=0.2, 
                        help="Fraction of data to use for testing")
    ## ^ Is this x% of the TOTAL data or of the [starting_update:final_idx] data? ...
    parser.add_argument('-kfcv', "--use_kfold_crossval", type=bool, default=True,
                        help="Split testing data by holding out some users (fraction held out determined by test_split_fraction)")
    parser.add_argument('-test_split_each_update', "--test_split_each_update", type=bool, default=False,
                        help="Implement train/test split within each update or on the entire dataset")
    parser.add_argument('-ts_ids', "--test_subj_IDs", type=str, default='[]',
                        help="List of subject ID strings of all subjects to be set to test only")
    parser.add_argument('-nkfs', "--num_kfold_splits", type=int, default=5,
                        help="Number of K Fold for Cross Validation")
    parser.add_argument('-maxkfs', "--num_max_kfold_splits", type=int, default=1,
                        help="Number of K Fold for Cross Validation")
    #
    # default=str(['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']),
    parser.add_argument('-lcidsq', "--live_client_IDs_queue", type=str, default='[]',
                        help="List of current subject ID strings (models will be trained and saved) --> THEY ARE QUEUED SO ONLY ONE WILL TRAIN AT A TIME")
    ## This needs to be changed to a smaller set (that does not include the future live clients) when training a model for Sequential
    ## Default (non-seq) --> Actually, I think this doesn't need to change. 
    ## This is the list of all clients it will train, both live and static. If not present here they are not used in training at all
    ## It just takes longer to load more data if you leave it as its default
    #
    parser.add_argument('-allsids', "--all_subj_IDs", type=str, default=str(['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']),
                        help="Subject ID Codes for users")

    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cpu",  # KAI: Changed the default to cpu
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cphs")  # KAI: Changed the default to cphs (from mnist)    
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')

    # SECTION: Idk what these are lol
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    #parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    #############################################################################################################
    parser.add_argument('-nnc', "--new_clients_ID_lst", type=str, default=str([]))
    # KAI: ^ I think that was for PFL algos, testing them on new clients. Seq replaces this...
    #############################################################################################################
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    
    # SECTION: practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threshold", type=float, default=10000,
                        help="The threshold for droping slow clients")
    parser.add_argument('-lth', "--loss_threshold", type=int, default=10000,
                        help="The max loss threshold for aborting a training run")
    
    # SECTION: pFedMe / PerAvg
    parser.add_argument('-lam', "--lambda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    
    # SECTION: Kai's additional args
    # Idk if this one is even used/useful anymore (think it was for interfacing with ipynb?)
    parser.add_argument('-run', "--run", type=bool, default=True,
                        help="If False, will set up the arg parser and args variable, but won't run")
    parser.add_argument('-scll', "--save_client_loss_logs", type=bool, default=True,
                        help="Boolean determing whether or not to save each clients testing loss log")
    parser.add_argument('-taaoc', "--test_against_all_other_clients", type=bool, default=False,
                        help="Boolean for whether or not to test each client's model on all other clients. As on 11/26 only supported for ServerLocal")
    parser.add_argument('-ccm', "--cross_client_modulus", type=int, default=5,
                        help="Number of rounds between cross client testing (current_round%cross_client_modulus==0)")
    # PCA should probably be broken into 2 since 64 channels is device specific
    parser.add_argument('-pca_ch', "--pca_channels", type=int, default=64,
                        help="Number of principal components. 64 means do not use any PCA")
    parser.add_argument('-lF', "--lambdaF", type=float, default=0.0,
                        help="Penalty term for user EMG input (user effort)")
    parser.add_argument('-lD', "--lambdaD", type=float, default=1e-4,
                        help="Penalty term for the decoder norm (interface effort)")
    parser.add_argument('-lE', "--lambdaE", type=float, default=1e-6,
                        help="Penalty term on performance error norm")
    parser.add_argument('-sbb', "--smoothbatch_boolean", type=bool, default=True,
                        help="Boolean switch for whether or not to use SmoothBatch. See Madduri CPHS Paper.")
    parser.add_argument('-sblr', "--smoothbatch_learningrate", type=float, default=0.75, #0.75 is the slow condition, 0.25 is fast
                        help="Value of alpha (mixing param) for SB. Alpha=1 uses only the optimal dec, Alpha=0 uses only the previous dec")
    ##
    parser.add_argument('-device_ch', "--device_channels", type=int, default=64,
                        help="Number of recording channels with the used EMG device")
    parser.add_argument('-dt', "--dt", type=float, default=1/60,
                        help="Delta time, amount of time (sec?) between measurements")
    parser.add_argument('-normalize_data', "--normalize_data", type=bool, default=True,
                        help="Normalize (actually scales...) the input EMG signals and its labels. This is good practice.")
    parser.add_argument('-debug_mode', "--debug_mode", type=bool, default=False,
                        help="Will do additional checks on loss magnitudes and such (check_loss_for_nan_inf, etc).")
    parser.add_argument('-v', "--verbose", type=bool, default=False,
                        help="Print out a bunch of extra stuff")
    parser.add_argument('-slow_clients_bool', "--slow_clients_bool", type=bool, default=False,
                        help="Control whether or not to have ANY slow clients")
    parser.add_argument('-return_cost_func_comps', "--return_cost_func_comps", type=bool, default=True,  # They're basically returned by default now
                        help="Return Loss, Error, DTerm, FTerm from loss class")
    parser.add_argument('-lm_bias', "--linear_model_bias", type=bool, default=True,
                        help="Boolean determining whether to use an additive bias. Note that previous 599 approach had no additive bias.")
    # Idk if this will work actually? :0.self.ndpf won't work, I don't think...
    parser.add_argument('-ndp', "--num_decimal_points", type=int, default=5,
                        help="Number of decimal points to use when format printing.")
    # This one is not integrated yet
    parser.add_argument('-rtm', "--run_train_metrics", type=bool, default=True,
                        help="Evaluate every client on the training data")  # I don't think this matters for local, since every client is being run anyways?
    parser.add_argument('-ubl', "--update_batch_length", type=int, default=1200,
                        help="Minimum length of the simulated streamed updates (in CPHS, was 1200).") 
    
    # THIS IS ALL OLD COMPARED TO MAIN_SEQ.PY, USE THAT INSTEAD
    ## SEQUENTIAL TRAINING PARAMS
    parser.add_argument('-seq', "--sequential", type=bool, default=False,
                        help="Boolean toggle for whether sequential mode is on (for now, mixing current client with previously trained models)")
    parser.add_argument('-uppm', "--use_prev_pers_model", type=bool, default=False,
                        help="Boolean toggle for whether to use previously trained personalized models for the client inits")
    parser.add_argument('-nlsrpsq', "--num_liveseq_rounds_per_seqclient", type=int, default=25,
                        help="Number of training rounds to do in a row on a single live (seq) client before advancing to the next seq client.")    
    parser.add_argument('-scids', "--static_client_IDs", type=str, default='[]',
                        help="List of previously trained subject ID strings (models will be uploaded, used in training, but never updated)")
    parser.add_argument('-svlweight', "--static_vs_live_weighting", type=float, default=0.75,
                        help="Ratio between number of static clients and live clients present in each training round. Set completely arbitrarily for now.")
    ## For FedAvg all clients have the same model so code has to change to reflect this
    ### Current saving regime is broken and saves Local correctly I believe but not FedAvg (no FedAvg directory even...)
    #### My default entry is the path with Latest FedAvg filename, despite what the help description says...
    parser.add_argument('-pmd', "--prev_model_directory", type=str, default="C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Personalized_Federated_Learning\\models\\cphs\\FedAvg\\11-10_16-14\\FedAvg_server_global.pt",
                        help="Directory name containing all the prev clients models") 
    # This needs to be updated...
    #parser.add_argument('-ppmd', "--prev_pers_model_directory", type=str, default="C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Personalized_Federated_Learning\\models\\cphs\\FedAvg_server.pt",
    #                    help="Directory name containing all the prev clients personalized models") 

    #############################################################################################################################################

    args = parser.parse_args()

    args.condition_number_lst = convert_cmd_line_str_lst_to_type_lst(args.condition_number_lst, int)
    args.all_subj_IDs = convert_cmd_line_str_lst_to_type_lst(args.all_subj_IDs, str)
    if args.test_subj_IDs!=[]:
        args.test_subj_IDs = convert_cmd_line_str_lst_to_type_lst(args.test_subj_IDs, str)
    #if args.sequential != False:
    args.live_client_IDs_queue = convert_cmd_line_str_lst_to_type_lst(args.live_client_IDs_queue, str)
    args.static_client_IDs = convert_cmd_line_str_lst_to_type_lst(args.static_client_IDs, str)
    
    # I always need to run on CPU only since I don't have Nvidia GPU available
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    return args


if __name__ == "__main__":
    total_start = time.time()

    # Idk if this will work with command line now or not lol
    args = parse_args()
    
    print("=" * 50)

    if args.run!=True:
        print("YOU HAVE NOT TOGGLED ARGS.RUN TO BE TRUE AND THUS THIS WILL NOT RUN")
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("train_subj_IDs subjects: {}".format(args.all_subj_IDs))
    print("List of all condition numbers to train over: {}".format(args.condition_number_lst))
    #print("Local learing rate decay: {}".format(args.learning_rate_decay))
    #if args.learning_rate_decay:
    #    print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(len(args.all_subj_IDs)))
    print("Clients join in each round: {}".format(args.join_ratio))
    #print("Clients randomly join: {}".format(args.random_join_ratio))
    #print("Client drop rate: {}".format(args.client_drop_rate))
    #print("Client select regarding time: {}".format(args.time_select))
    #if args.time_select:
    #    print("Time threshold: {}".format(args.time_threshold))
    #print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Backbone (model): {}".format(args.model_str))
    #print("Using device: {}".format(args.device))
    #print("Using DP: {}".format(args.privacy))
    #if args.privacy:
    #    print("Sigma for DP: {}".format(args.dp_sigma))
    #print("Auto break: {}".format(args.auto_break))
    #if not args.auto_break:
    print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    ############################################################################################################################################
    print("Total number of new clients: {}".format(len(args.new_clients_ID_lst)))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    
    print()
    print("KAI'S ADDITIONS")
    if args.pca_channels!=64:
        print("Number of PCA Components Used: {}".format(args.pca_channels))
    print(f"Lambda penalty terms (F, D, E): {args.lambdaF}, {args.lambdaD}, {args.lambdaE}")
    print("Starting update: {}".format(args.starting_update))
    # TODO: REWORK THIS!!!
    print("Testing split: {}".format(args.test_split_fraction))
    if args.dt!=1/60:
        print("dt: {}".format(args.dt))
    print("Normalize data: {}".format(args.normalize_data))
    print("Local round threshold: {}".format(args.local_round_threshold))
    print("In Debug Mode: {}".format(args.debug_mode))
    
    print("=" * 50)

    print()
    print(f"YOU ARE RUNNING -{args.algorithm}- ALGORITHM")

    if args.run:
        #server_obj = run(args)
        server_obj = run_kfcv(args)
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")