#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging

# Run setup.py so that you can import different modules
#os.popen('setup.sh')
# https://stackoverflow.com/questions/42900259/running-sh-file-from-python-script/42900528
# Is path required?  This one is in the same directory so I think it's fine
# ^ It runs but it runs in parallel so that the Python outpaces it... that seems wrong to me...
# Also only needs to be run on start up, not each time you run main

from flcore.servers.serveravg import FedAvg
#from flcore.servers.serverpFedMe import pFedMe  # Didn't save this file, need to retrieve it from fork
#from flcore.servers.serverperavg import PerAvg  # Didn't save this file, need to retrieve it from fork
from flcore.servers.serverlocal import Local
#from flcore.servers.serverper import FedPer  # Didn't save this file, need to retrieve it from fork
from flcore.servers.serverapfl import APFL
#from flcore.servers.serverscaffold import SCAFFOLD

from flcore.pflniid_utils.result_utils import average_data
from flcore.pflniid_utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "LinearRegression":
            args.model = torch.nn.Linear(args.pca_channels, 2)  #input_size, output_size
        else:
            raise NotImplementedError

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
        #elif args.algorithm == "PerAvg":
        #    server = PerAvg(args, i)
        elif args.algorithm == "APFL":
            server = APFL(args, i)
        #elif args.algorithm == "FedPer":
        #    # FIX ARGS.HEAD --> LINEAR REGRESSION HAS NO FC
        #    #args.head = copy.deepcopy(args.model.fc)
        #    #args.model.fc = nn.Identity()
        #    #args.model = BaseHeadSplit(args.model, args.head)
        #    server = FedPer(args, i)
        #elif args.algorithm == "SCAFFOLD":
        #    server = SCAFFOLD(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
      
    # Global average
    #average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    # Idk what that is supposed to do.  Prints acc from some file.  Don't need it, acc isn't the same for my task
    print("Server's rs_train_loss: ")
    print(server.rs_train_loss)  # I think this is a list...

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cpu",  # KAI: Changed the default to cpu
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cphs")  # KAI: Changed the default to cphs (from mnist)
    #parser.add_argument('-nb', "--num_classes", type=int, default=10)  # Not doing classification...
    parser.add_argument('-m', "--model", type=str, default="LinearRegression")  # KAI: Changed the default to Linear Regression
    parser.add_argument('-lbs', "--batch_size", type=int, default=1200)  # Setting it to a full update would be 1300ish... how many batches does it run? In one epoch? Not even sure where that is set
    # The 1300 and the batch size are 2 separate things...
    # I want to restrict the given dataset to just the 1300, but then iterate in batches... or do I since we don't have that much data and can probably just use all the data at once? Make batch size match the update size? ...
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)  # KAI: Switched to 50 down from 2000
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")  # KAI: I think it was 1 originally.  I'm gonna keep it there.  Does this mean I can set batchsize to 1300 and cook?Is my setup capable or running multiple epochs? Implicitly I was doing 1 epoch before, using the full update data I believe...
    parser.add_argument('-algo', "--algorithm", type=str, default="Local")#default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.2,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=14,
                        help="Total number of clients")
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
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)  # DLG = Deep Leakage from Gradients
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)  # Only used with DLG
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
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
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    
    # SECTION: pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    
    # SECTION: Kai's additional args
    parser.add_argument('-pca_channels', "--pca_channels", type=int, default=64,
                        help="Number of principal components. 64 means do not use any PCA")
    parser.add_argument('-lambdaF', "--lambdaF", type=float, default=0.0,
                        help="Penalty term for user EMG input (user effort)")
    parser.add_argument('-lambdaD', "--lambdaD", type=float, default=1e-3,
                        help="Penalty term for the decoder norm (interface effort)")
    parser.add_argument('-lambdaE', "--lambdaE", type=float, default=1e-4,
                        help="Penalty term on performance error norm")
    parser.add_argument('-starting_update', "--starting_update", type=int, default=0,
                        help="Which update to start on (for CPHS Simulation). Use 0 or 10.")
    parser.add_argument('-test_split_fraction', "--test_split_fraction", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument('-device_channels', "--device_channels", type=int, default=64,
                        help="Number of recording channels with the used EMG device")
    parser.add_argument('-dt', "--dt", type=float, default=1/60,
                        help="Delta time, amount of time (sec?) between measurements")
    parser.add_argument('-normalize_emg', "--normalize_emg", type=bool, default=False,
                        help="Normalize the input EMG signals")
    parser.add_argument('-normalize_V', "--normalize_V", type=bool, default=False,
                        help="Normalize the V term in the cost function")
    parser.add_argument('-local_round_threshold', "--local_round_threshold", type=int, default=50,
                        help="Number of communication rounds per client until a client will advance to the next batch of streamed data")
    parser.add_argument('-debug_mode', "--debug_mode", type=bool, default=False,
                        help="In debug mode, the code is run to minimize overhead time in order to debug as fast as possible.  Namely, the data is held at the server to decrease init time, and communication delays are ignored.")
    parser.add_argument('-condition_number', "--condition_number", type=int, default=1,
                        help="Which condition number (trial) to train on")
    parser.add_argument('-test_split_each_update', "--test_split_each_update", type=bool, default=False,
                        help="Implement train/test split within each update or on the entire dataset")
    
    args = parser.parse_args()

    # I always need to run on CPU only since I don't have Nvidia GPU available
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    #print("Local learing rate decay: {}".format(args.learning_rate_decay))
    #if args.learning_rate_decay:
    #    print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    #print("Clients randomly join: {}".format(args.random_join_ratio))
    #print("Client drop rate: {}".format(args.client_drop_rate))
    #print("Client select regarding time: {}".format(args.time_select))
    #if args.time_select:
    #    print("Time threthold: {}".format(args.time_threthold))
    #print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    #print("Number of classes: {}".format(args.num_classes))
    print("Backbone (model): {}".format(args.model))
    #print("Using device: {}".format(args.device))
    #print("Using DP: {}".format(args.privacy))
    #if args.privacy:
    #    print("Sigma for DP: {}".format(args.dp_sigma))
    #print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    #print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    
    print()
    print("KAI'S ADDITIONS")
    if args.pca_channels!=64:
        print("Number of PCA Components Used: {}".format(args.pca_channels))
    print(f"Lambda penalty terms (F, D, E): {args.lambdaF}, {args.lambdaD}, {args.lambdaE}")
    print("Starting update: {}".format(args.starting_update))
    print("Testing split: {}".format(args.test_split_fraction))
    if args.dt!=1/60:
        print("dt: {}".format(args.dt))
    print("Normalize EMG input: {}".format(args.normalize_emg))
    print("Normalize V term: {}".format(args.normalize_V))
    print("Local round threshold: {}".format(args.local_round_threshold))
    print("In Debug Mode: {}".format(args.debug_mode))
    
    print("=" * 50)

    print()
    print(f"YOU ARE RUNNING -{args.algorithm}- ALGORITHM")
    #print()

    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")