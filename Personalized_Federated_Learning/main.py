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

import argparse
import os
import time
import warnings
import logging
from utils.helper_funcs import convert_cmd_line_str_lst_to_type_lst

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
#from flcore.servers.serverper import FedPer
#from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto

'''
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
'''

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
            args.model = torch.nn.Linear(args.pca_channels, 2, args.linear_model_bias)  #input_size, output_size
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
        elif args.algorithm == "APFL":
            server = APFL(args, i)
        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)
        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)
        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)
        #elif args.algorithm == "FedPer":
        #    #args.head = copy.deepcopy(args.model.fc)
        #    #args.model.fc = nn.Identity()
        #    #args.model = BaseHeadSplit(args.model, args.head)
        #    server = FedPer(args, i)
        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        #elif args.algorithm == "SCAFFOLD":
        #    server = SCAFFOLD(args, i)
        else:
            raise NotImplementedError
        
        '''
        elif args.algorithm == "FedPer":
            # FIX ARGS.HEAD --> LINEAR REGRESSION HAS NO FC
            #args.head = copy.deepcopy(args.model.fc)
            #args.model.fc = nn.Identity()
            #args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)
        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)
        elif args.algorithm == "FedProx":
            server = FedProx(args, i)
        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)
        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)
        elif args.algorithm == "APFL":
            server = APFL(args, i)
            elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)
        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)
        elif args.algorithm == "FedBN":
            server = FedBN(args, i)
        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)
        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)
        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)
        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)
        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)
        elif args.algorithm == "APPLE":
            server = APPLE(args, i)
        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)
            elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)
        elif args.algorithm == "FedALA":
            server = FedALA(args, i)
        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)
        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)
        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)
        elif args.algorithm == "FML":
            server = FML(args, i)
        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)
        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)
        '''

        server.train()
        time_list.append(time.time()-start)

    server.plot_results()
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
      
    # Global average
    if args.algorithm != "Local":
        average_data(server.trial_result_path, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
        # Not super sure what "times" is, by default it is 1. Assuming it runs the process multiple times to average out the stochasticity?

    # Idk why I was printing this...  
    #print("Server's round, rs_train_loss, rs_test_loss (averaged over clients): ")
    #print(server.rs_train_loss)  # I think this is a list...
    #print("Server's rs_test_loss (averaged over clients): ")
    #print(server.rs_test_loss)
    #assert( len(server.rs_train_loss) == len(server.rs_test_loss))
    #if args.run_train_metrics:
    #    for i in range(len(server.rs_train_loss)):
    #        print(f"Round {i}, Train Loss: {server.rs_train_loss[i]:0.5f}, Test Loss: {server.rs_test_loss[i]:0.5f}")
    #        if i==(len(server.rs_train_loss)-1):
    #            print(f"Final eval ({i+1}), Test Loss: {server.rs_test_loss[i+1]:0.5f}")
    #else:
    #    for i in range(len(server.rs_test_loss)):
    #        print(f"Round {i}, Test Loss: {server.rs_test_loss[i]:0.5f}")

    print("All done!")
    reporter.report()
    return server


def parse_args():
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
    # I have little confidence in this batch size being correct...
    parser.add_argument('-lbs', "--batch_size", type=int, default=1202)  # Setting it to a full update would be 1202... will this automatically run twice?
    # The 1300 and the batch size are 2 separate things...
    # I want to restrict the given dataset to just the 1300, but then iterate in batches... or do I since we don't have that much data and can probably just use all the data at once? Make batch size match the update size? ...
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=1,  #0.005
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)  # KAI: Originally was 2000
    parser.add_argument('-ls', "--local_epochs", type=int, default=3, 
                        help="Multiple update steps in one local epoch.")  # KAI: I think it was 1 originally.  I'm gonna keep it there.  Does this mean I can set batchsize to 1300 and cook? Is my setup capable or running multiple epochs? Implicitly I was doing 1 epoch before, using the full update data I believe...
    #Local #FedAvg #APFL #FedMTL #pFedMe ## #Ditto #PerAvg
    ## pFedMe not working
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.3,
                        help="Fraction of clients to be active in training per round")
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
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)  # DLG = Deep Leakage from Gradients
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=None)  # Only used with DLG
    #############################################################################################################
    parser.add_argument('-nnc', "--new_clients_ID_lst", type=str, default='[]')
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
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    
    '''
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    '''
    
    # SECTION: Kai's additional args
    parser.add_argument('-run', "--run", type=bool, default=True,
                        help="If False, will set up the arg parser and args variable, but won't run")
    # PCA should probably be broken into 2 since 64 channels is device specific
    parser.add_argument('-pca_ch', "--pca_channels", type=int, default=10,
                        help="Number of principal components. 64 means do not use any PCA")
    parser.add_argument('-lF', "--lambdaF", type=float, default=0.0,
                        help="Penalty term for user EMG input (user effort)")
    parser.add_argument('-lD', "--lambdaD", type=float, default=1e-3, #1e-3
                        help="Penalty term for the decoder norm (interface effort)")
    parser.add_argument('-lE', "--lambdaE", type=float, default=1e-4, #1e-4
                        help="Penalty term on performance error norm")
    parser.add_argument('-stup', "--starting_update", type=int, default=10,
                        help="Which update to start on (for CPHS Simulation). Use 0 or 10.")
    ## Test Split Related
    parser.add_argument('-test_split_fraction', "--test_split_fraction", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument('-test_split_each_update', "--test_split_each_update", type=bool, default=False,
                        help="Implement train/test split within each update or on the entire dataset")
    parser.add_argument('-test_split_users', "--test_split_users", type=bool, default=False,
                        help="Split testing data by holding out some users (fraction held out determined by test_split_fraction)")
    parser.add_argument('-ts_ids', "--test_subj_IDs", type=str, default='[]',
                        help="List of subject ID strings of all subjects to be set to test only")
    ##
    parser.add_argument('-device_ch', "--device_channels", type=int, default=64,
                        help="Number of recording channels with the used EMG device")
    parser.add_argument('-dt', "--dt", type=float, default=1/60,
                        help="Delta time, amount of time (sec?) between measurements")
    parser.add_argument('-normalize_data', "--normalize_data", type=bool, default=True, # Only works when True!
                        help="Normalize the input EMG signals and its labels. This is good practice.")
    parser.add_argument('-lrt', "--local_round_threshold", type=int, default=50,
                        help="Number of communication rounds per client until a client will advance to the next batch of streamed data")
    # I think I depreciated debug_mode, double check it's removed
    parser.add_argument('-debug_mode', "--debug_mode", type=bool, default=False,
                        help="I THINK I KILLED THIS MODE: In debug mode, the code is run to minimize overhead time in order to debug as fast as possible.  Namely, the data is held at the server to decrease init time, and communication delays are ignored.")
    parser.add_argument('-con_num', "--condition_number_lst", type=str, default='[1]',
                        help="Which condition number (trial) to train on. Must be a list. By default, will iterate through all train_subjs for each cond (eg each cond_num gets its own client even for the same subject)")
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
    ## SEQUENTIAL TRAINING PARAMS
    parser.add_argument('-seq', "--sequential", type=bool, default=False,
                        help="Boolean toggle for whether sequential mode is on (for now, mixing current client with previously trained models)")
    parser.add_argument('-uppm', "--use_prev_pers_model", type=bool, default=False,
                        help="Boolean toggle for whether to use previously trained personalized models for the client inits")
    # default=str(['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']),
    parser.add_argument('-live_clients', "--live_clients", type=str, default='[]',
                        help="List of current subject ID strings (models will be trained and saved)")
    parser.add_argument('-static_clients', "--static_clients", type=str, default='[]',
                        help="List of previously trained subject ID strings (models will be uploaded, used in training, but never updated)")
    parser.add_argument('-svlweight', "--static_vs_live_weighting", type=float, default=0.75,
                        help="Ratio between number of static clients and live clients present in each training round. Set completely arbitrarily for now.")
    # This needs to be changed to a smaller set (that does not include the future live clients) when training a model for Sequential
    ## Default (non-seq) --> Actually, I think this doesn't need to change. 
    ### This is the list of all clients it will train, both live and static. If not present here they are not used in training at all
    ### It just takes longer to load more data if you leave it as its default
    parser.add_argument('-tr_ids', "--train_subj_IDs", type=str, default=str(['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']),
                        help="Subject ID Codes for users to be trained")
    ## For FedAvg all clients have the same model so code has to change to reflect this
    ### Current saving regime is broken and saves Local correctly I believe but not FedAvg (no FedAvg directory even...)
    #### My default entry is the path with Latest FedAvg filename, despite what the help description says...
    parser.add_argument('-pmd', "--prev_model_directory", type=str, default="C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Personalized_Federated_Learning\\models\\cphs\\FedAvg\\11-07_11-31\FedAvg_server_global.pt",
                        help="Directory name containing all the prev clients models") 
    # This needs to be updated...
    #parser.add_argument('-ppmd', "--prev_pers_model_directory", type=str, default="C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Personalized_Federated_Learning\\models\\cphs\\FedAvg_server.pt",
    #                    help="Directory name containing all the prev clients personalized models") 

    #############################################################################################################################################

    args = parser.parse_args()

    args.condition_number_lst = convert_cmd_line_str_lst_to_type_lst(args.condition_number_lst, int)
    args.train_subj_IDs = convert_cmd_line_str_lst_to_type_lst(args.train_subj_IDs, str)
    if args.test_subj_IDs!=[]:
        args.test_subj_IDs = convert_cmd_line_str_lst_to_type_lst(args.test_subj_IDs, str)
    if args.sequential != False:
        args.live_clients = convert_cmd_line_str_lst_to_type_lst(args.live_clients, str)
        args.static_clients = convert_cmd_line_str_lst_to_type_lst(args.static_clients, str)
    
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
    print("Subjects in training data: {}".format(args.train_subj_IDs))
    print("List of all condition numbers to train over: {}".format(args.condition_number_lst))
    #print("Local learing rate decay: {}".format(args.learning_rate_decay))
    #if args.learning_rate_decay:
    #    print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(len(args.train_subj_IDs)))
    print("Clients join in each round: {}".format(args.join_ratio))
    #print("Clients randomly join: {}".format(args.random_join_ratio))
    #print("Client drop rate: {}".format(args.client_drop_rate))
    #print("Client select regarding time: {}".format(args.time_select))
    #if args.time_select:
    #    print("Time threshold: {}".format(args.time_threshold))
    #print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
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
    ############################################################################################################################################
    print("Total number of new clients: {}".format(len(args.new_clients_ID_lst)))
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
    print("Normalize data: {}".format(args.normalize_data))
    print("Local round threshold: {}".format(args.local_round_threshold))
    print("In Debug Mode: {}".format(args.debug_mode))
    
    print("=" * 50)

    print()
    print(f"YOU ARE RUNNING -{args.algorithm}- ALGORITHM")

    if args.run:
        server_obj = run(args)
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")