# PFLNIID

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from datetime import datetime
import matplotlib.pyplot as plt
from math import ceil


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset #Really ought to be dataset_str...
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.local_learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.personalized_algorithms = ["APFL", "FedMTL", "PerAvg", "PerFedAvg", "pFedMe", "FedPer", "Ditto"]
        self.personalized_algo_bool = True if self.algorithm.upper() in [algo.upper() for algo in self.personalized_algorithms] else False
        self.time_select = args.time_select  # What is this...
        self.goal = args.goal  # This can probably be removed entirely...
        self.time_threshold = args.time_threshold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 20  # What is ths...
        #self.auto_break = args.auto_break  # What is ths...

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        # This is used in recieve_models(), not super sure for what (or if I care)
        self.uploaded_weights = []
        self.uploaded_IDs = []
        self.uploaded_models = []

        self.save_client_loss_logs = args.save_client_loss_logs
        self.save_models = args.save_models
        self.rs_test_loss = []
        self.rs_train_loss = []
        # Can't save dicts to HD5F files so use nested lists for now I guess
        self.cost_func_comps_log = []
        self.gradient_norm_log = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.new_clients_ID_lst = args.new_clients_ID_lst
        self.num_new_clients = len(self.new_clients_ID_lst)
        self.new_clients_obj_lst = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        
        # Kai's additional params
        self.smoothbatch_boolean = args.smoothbatch_boolean
        self.smoothbatch_learningrate = args.smoothbatch_learningrate
        self.models_base_dir = "models"
        self.model_str = args.model_str
        if self.model_str != "LinearRegression":
            self.deep_bool = True
        else:
            self.deep_bool = False
        self.ndp = args.num_decimal_points
        self.global_round = 0
        self.debug_mode = args.debug_mode
        self.starting_update = args.starting_update
        self.global_update = args.starting_update # Where is global_update even used lol
        self.verbose = args.verbose
        self.slow_clients_bool = args.slow_clients_bool
        self.run_train_metrics = args.run_train_metrics
        # Absolute path
        #self.result_path = "C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Personalized_Federated_Learning\\results\\"
        # Relative path
        self.result_path = "\\results\\" 
        # serverlocal ONLY!
        self.test_against_all_other_clients = args.test_against_all_other_clients
        self.cross_client_modulus = args.cross_client_modulus
        # Deep learning stuff
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.sequence_length = args.sequence_length
        self.output_size = args.output_size
        # Not used on server but saved when logging
        self.pca_channels = args.pca_channels
        self.device_channels = args.device_channels
        self.lambdaF = args.lambdaF
        self.lambdaD = args.lambdaD
        self.lambdaE = args.lambdaE
        self.normalize_data = args.normalize_data
        self.local_round_threshold = args.local_round_threshold
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        # Testing
        self.test_split_each_update = args.test_split_each_update
        self.test_subj_IDs = args.test_subj_IDs
        self.test_split_fraction = args.test_split_fraction
        self.use_kfold_crossval = args.use_kfold_crossval
        self.num_kfolds = args.num_kfold_splits
        self.num_max_kfold_splits = args.num_max_kfold_splits
        self.current_fold = 0
        # Trial set up
        self.condition_number_lst = args.condition_number_lst

        self.all_subj_IDs = args.all_subj_IDs
        self.train_subj_IDs = None

        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        #self.num_clients = 0  #len(self.train_subj_IDs) * len(self.condition_number_lst)  
        self.num_clients = len(self.all_subj_IDs) // self.num_kfolds + (1 if len(self.all_subj_IDs) % self.num_kfolds > 0 else 0)
        self.num_join_clients = ceil(self.num_clients * self.join_ratio)
        assert(self.num_join_clients>=1)
        if self.num_join_clients==1:
            print("num_join_clients IS JUST ONE (1) CLIENT!")
        self.loss_threshold = args.loss_threshold
        ## SEQUENTIAL TRAINING PARAMS
        self.sequential = args.sequential
        self.live_client_IDs_queue = args.live_client_IDs_queue
        self.live_clients = [] # Empty list. Should only ever have 1 client in it for SEQUENTIAL
        self.live_idx = 0
        self.num_liveseq_rounds_per_seqclient = args.num_liveseq_rounds_per_seqclient
        self.static_client_IDs = args.static_client_IDs
        self.static_vs_live_weighting = args.static_vs_live_weighting
        self.prev_model_directory = args.prev_model_directory
        self.use_prev_pers_model = args.use_prev_pers_model
        self.curr_live_rs_test_loss = []
        self.prev_live_rs_test_loss = []
        self.unseen_live_rs_test_loss = []
        self.unseen_live_client_IDs = copy.deepcopy(self.live_client_IDs_queue)
        self.prev_live_client_IDs = []
        #self.prev_pers_model_directory = args.prev_pers_model_directory

        # Dataset is one of the preceeding directory names
        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        self.str_current_datetime = str(current_datetime)
        # Get the directory of the current script
        self.script_directory = os.path.dirname(os.path.abspath(__file__))[:-15]  # This returns the path to serverbase... so don't index the end of the path
        # Specify the relative path from the script's directory
        #relative_path = os.path.join(self.result_path, self.str_current_datetime+"_"+self.algorithm)
        if self.sequential:
            seq_str = "_Seq_"
        else:
            seq_str = "_"
        if self.model_str=="LinearRegression":
            model_str = "LinRegr"
        else:
            model_str = self.model_str
        self.relative_path = self.result_path + self.str_current_datetime + seq_str + self.algorithm + "_" + model_str + str(self.global_rounds)

        self.ewc_bool = args.ewc_bool
        self.fisher_mult = args.fisher_mult
        self.optimizer_str = args.optimizer_str


    def create_client_mapping(self, clients):
            return {client.ID: client for client in clients}


    def set_clients(self, clientObj):  
        if self.verbose:
            print("ServerBase Set_Clients (SBSC) -- probably called in init() of server children classes")
        
        base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Data\\CPHS_EMG\\Subject_Specific_Files\\'
        #for i, train_slow, send_slow in zip(range(len(self.all_subj_IDs)), self.train_slow_clients, self.send_slow_clients):
        for i in range(len(self.all_subj_IDs)):
            for j in self.condition_number_lst:
                print(f"SB Set Client: iter {i}, cond number: {str(j)}: LOADING DATA: {self.all_subj_IDs[i]}")
                # For now, have to load the data because of how I set it up
                # Look into changing this in the future...
                ## This actually has to stay if I want to be able to run train/test_metrics() on the past clients
                ## ^Since those functions require the local data in order to eval the model
                ID_str = self.all_subj_IDs[i]
                client = clientObj(self.args, 
                                    ID=ID_str, 
                                    samples_path = base_data_path + 'S' + ID_str[-3:] + "_TrainData_8by20770by64.npy", 
                                    labels_path = base_data_path + 'S' + ID_str[-3:] + "_Labels_8by20770by2.npy", 
                                    condition_number = j-1, 
                                    train_slow=False, 
                                    send_slow=False)
                client.load_train_data(client_init=True) # This has to be here otherwise load_test_data() breaks... is this still true? I think so?

                #if self.sequential and (self.all_subj_IDs[i] in self.static_client_IDs):  
                if self.sequential: # Load the prev global model for all clients actually
                    print(f"SB Set Client: LOADING MODEL: {self.all_subj_IDs[i]}")
                    # Load the client model
                    ## This is not super robust. Assume the full path is the provided path and the file name is just the ID...
                    ### Does this need to be updated to include the file extension? Probably... .pt?
                    #path_to_trained_client_model = self.prev_model_directory + self.ID

                    if (self.use_prev_pers_model==True) and (self.all_subj_IDs[i] in self.static_client_IDs):
                        # Then load the previous personalized model
                        # path_to_trained_client_model = self.prev_pers_model_directory + self.ID
                        # model_name = "somethingsomething_server_pers.pt" # Presumably include client ID...
                        raise("This is not supported yet")
                    else:
                        # Else they do not have a personalized model already (eg fresh client)
                        # Thus default to the global model
                        path_to_trained_client_model = self.prev_model_directory
                        model_name = "FedAvg_server_global.pt"
                    # Need to print whether the personalized or locally fine-tuned model was used...
                    # Requires full path to model (eg with extension)
                    client.load_item(model_name, full_path_to_item=path_to_trained_client_model)
                
                self.clients.append(client)

        # Also create a mapping from subj_IDs to client objects
        self.dict_map_subjID_to_clientobj = self.create_client_mapping(self.clients)
                

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for _ in range(self.num_clients)]
        if self.slow_clients_bool==False:
            return slow_clients
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients


    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)


    def select_clients(self):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        
        if self.sequential:
            # Select first client in live_client_IDs_queue if this is the first round
            if self.global_round==0: 
                print(f"SB sel_cli: Global round {self.global_round}: setting first live client")
                # List of client objects which match the current live_indices (presumably live_idx=0)
                self.live_clients = [client_obj for client_obj in self.clients if client_obj.ID==self.live_client_IDs_queue[self.live_idx]]

                # Remove the new live client from the unseen log
                for c in self.live_clients:
                    if c.ID in self.unseen_live_client_IDs:
                        self.unseen_live_client_IDs.remove(c.ID)

            elif self.global_round%self.num_liveseq_rounds_per_seqclient==0:
                # ^Check if that client has been trained enough to switch

                # Now increment the client index
                self.live_idx += 1
                if self.live_idx == len(self.live_client_IDs_queue):
                    self.live_idx = 0
                # Put the current client in the prev_cli lst
                ## First check that they are not already in the lst
                finished_cli_ID = self.live_clients[0].ID
                if finished_cli_ID not in self.prev_live_client_IDs:
                    self.prev_live_client_IDs.append(finished_cli_ID)
                # Now select the next seq live client
                self.live_clients = [client_obj for client_obj in self.clients if client_obj.ID==self.live_client_IDs_queue[self.live_idx]]

                # Remove the new live client from the unseen log
                for c in self.live_clients:
                    if c.ID in self.unseen_live_client_IDs:
                        self.unseen_live_client_IDs.remove(c.ID)

                #if verbose:
                print()
                print("UPDATING TO NEW CLIENT!!!")
                print(f"unseen_live_client_IDs: {self.unseen_live_client_IDs}")
                print(f"live_clients: {self.live_clients}")  # [0].ID <-- Idk why that was there...
                print(f"prev_live_client_IDs: {self.prev_live_client_IDs}")
                print()

            # This is only true for the single user "seq" case...  
            assert(len(self.live_clients)==1)
            
            if num_join_clients > len(self.live_clients):
                #remaining_client_ids = [client.ID for client in self.clients if client.ID not in self.live_clients]
                # ^I already have this actually, it's just self.static_client_IDs. Might be better to do implicitly tho...
                random.shuffle(self.static_client_IDs) # Idk if I should be doing this honestly lol, cause idxs might get mixed up
                selected_clients = [client_obj for client_obj in self.clients if client_obj.ID in self.static_client_IDs[:(num_join_clients - len(self.live_clients))]]
                # Add in the single live client
                selected_clients.extend(self.live_clients)
            elif num_join_clients==1:
                # Only using live client, this is fine. Just be conscious you are making this choice
                selected_clients = self.live_clients
            else: #There are more live clients than total clients per round
                raise ValueError(f"More live clients ({len(self.live_clients)}) than allowed clients per round ({num_join_clients})")
                # This case isn't important right now. Ignore the below for now, maybe come back to it
                ## Thus only use the live clients for training? That relegates the rest of the clients to just pretraining...
                ## I would assume that we would get drift...
                if self.global_round<2:
                    print("SB: len(self.live_clients) > num_join_clients, so just sample the live clients")
                selected_clients = list(np.random.choice(self.live_clients, num_join_clients, replace=False))
        else:
            selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))
        return selected_clients


    def send_models(self):
        if self.verbose:
            print("SENDING GLOBAL MODEL TO CLIENTS")
        assert (len(self.clients) > 0)

        # The way they have it implies that EVERY CLIENT SETS TO THE GLOBAL MODEL EVERY ROUND...
        ## Go back and change this after the rest just to make sure it doesn't break the rest of the code
        # for client in self.clients: # This was the original
        for client in self.selected_clients: # This is my updated version...
            # If (seq is off) or (current client is the live seq client) then train as normal
            if (self.sequential==False) or ((self.sequential==True) and (client in self.live_clients)):
                start_time = time.time()
                
                client.set_parameters(self.global_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            # If seq is on but you are a static client, your model shouldn't be overwritten, thus do not send
            

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # This builds in clients dropping, which I have turned off by default
        active_clients = random.sample(
            self.selected_clients, ceil((1-self.client_drop_rate) * self.num_join_clients))

        self.uploaded_IDs = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            # Idk why they only consider clients below a time threshold... trying to ignore slow clients?
            if client_time_cost <= self.time_threshold:
                tot_samples += client.train_samples
                self.uploaded_IDs.append(client.ID)
                # This is w later (the weighting), right now is n_k
                self.uploaded_weights.append(client.train_samples)
                # This is the actual model (and params inside of it)
                self.uploaded_models.append(client.model)
            # Update client's last global round that it was included (eg to this round)
            client.last_global_round = self.global_round
        for i, w in enumerate(self.uploaded_weights):
            # This converts w to be n_k/N
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # Make a deep copy of the received model
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # Set global model params to zero
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            # Eg for each weighting and model pair that were uploaded:
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w


    '''
    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
    '''


    def load_model(self, directory_name, type):
        # This is not used, just fundamentally not how my saves work...
        '''
            Loads the specified model to become the current global model!

            str directory_name: name of when the model was saved, likely in the form of %m-%d_%H-%M unless it was renamed (this is the model's directory)
            str type: Should be one of 'global', 'pers', or 'local'
        '''
        model_path = os.path.join(self.models_base_dir, self.dataset, directory_name, self.algorithm + "_server_" + type + ".pt")
        # ^^ This really ought to be set somehow...
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    #def model_exists(self, directory_name, type):
    #    '''
    #        str directory_name: name of when the model was saved, likely in the form of %m-%d_%H-%M unless it was renamed (this is the model's directory)
    #        str type: Should be one of 'global', 'pers', or 'local'
    #   '''
    #    model_path = os.path.join(self.models_base_dir, self.dataset, directory_name, self.algorithm + "_server_" + type + ".pt")
    #    return os.path.exists(model_path)
    
        
    def save_results(self, personalized=False, save_cost_func_comps=False, save_gradient=False):
        # Combine the script's directory and the relative path to get the full path
        ## kfold_suffix must be updated since there is a new fold idx, which affects the rest of these file names...
        self.kfold_suffix = f"_kfold{self.current_fold}" if self.use_kfold_crossval else ""
        self.trial_result_path = self.script_directory+self.relative_path + self.kfold_suffix
        algo = self.algorithm + "_" + self.goal# + "_" + str(self.times) 
        self.h5_file_path = os.path.join(self.trial_result_path, "{}.h5".format(algo))
        self.paramtxt_file_path = os.path.join(self.trial_result_path, "param_log.txt")
        if not os.path.exists(self.trial_result_path):
            os.makedirs(self.trial_result_path)

        if self.save_models==True:
            # Save Server Global Model
            self.model_dir_path = os.path.join(self.models_base_dir, self.dataset, self.algorithm, self.str_current_datetime)
            if not os.path.exists(self.model_dir_path):
                os.makedirs(self.model_dir_path)
            model_file_path = os.path.join(self.model_dir_path, self.algorithm + "_server_global" + self.kfold_suffix + ".pt")
            torch.save(self.global_model, model_file_path)

            # Save client's local/personalized models (local and pers are the same objects)
            if self.personalized_algo_bool:
                client_algo_type = "Pers"
            else:
                client_algo_type = "Local"
            client_model_path = os.path.join(self.models_base_dir, self.dataset, client_algo_type, self.str_current_datetime)
            for client in self.clients:
                client.save_item(client.model, "local_client_model"+self.kfold_suffix, item_path=client_model_path)

            if personalized==True:
                pers_model_file_path = os.path.join(self.model_dir_path, self.algorithm + "_client_pers_model")
                for c in self.clients:
                    if not os.path.exists(pers_model_file_path):
                        print(f"SB pers model save made directory! {pers_model_file_path}")
                        os.makedirs(pers_model_file_path)
                    torch.save(c.model, os.path.join(self.model_dir_path, self.algorithm + "_client_pers_model", c.ID + "_pers_model" + self.kfold_suffix + ".pt"))

        sequential_base_list = [
            "\n\nSEQUENTIAL\n",
            f"sequential = {self.sequential}\n"]
        if self.sequential:
            sequential_base_list.extend([
                f"live_client_IDs_queue = {self.live_client_IDs_queue}\n",
                f"static_client_IDs = {self.static_client_IDs}\n",
                f"num_liveseq_rounds_per_seqclient = {self.num_liveseq_rounds_per_seqclient}\n",
                f"prev_model_directory = {self.prev_model_directory}"])
        sequential_base_str = ''.join(sequential_base_list)
        
        federated_base_str = ("\n\nFEDERATED LEARNING PARAMS\n"
            f"local_round_threshold = {self.local_round_threshold}\n")

        cphs_base_str = ("\n\nSIMULATION PARAMS\n"
            f"starting_update = {self.starting_update}\n"
            f"all_subj_IDs = {self.all_subj_IDs}\n"
            f"condition_number_lst = {self.condition_number_lst}\n"
            f"total effective clients = {self.num_clients}\n"
            # ^ Previously was calculated as: all_subj_IDs*condition_number_lst 
            ## But num_clients is not calculated that way anymore......
            f"smoothbatch_boolean = {self.smoothbatch_boolean}\n"
            f"smoothbatch_learningrate = {self.smoothbatch_learningrate}\n")

        param_log_str = (
            "BASE\n"
            f"algorithm = {self.algorithm}\n"
            f"model = {self.global_model}\n"
            f"device_channels = {self.device_channels}\n"
            "\n\nMODEL HYPERPARAMETERS\n"
            f"lambdaF = {self.lambdaF}\n"
            f"lambdaD = {self.lambdaD}\n"
            f"lambdaE = {self.lambdaE}\n"
            f"global_rounds = {self.global_rounds}\n"
            f"local_epochs = {self.local_epochs}\n"
            f"batch_size = {self.batch_size}\n"
            f"local_learning_rate = {self.local_learning_rate}\n"
            f"learning_rate_decay = {self.learning_rate_decay}\n"
            f"learning_rate_decay_gamma = {self.learning_rate_decay_gamma}\n"
            f"optimizer = {self.optimizer_str}\n"
            f"pca_channels = {self.pca_channels}\n"
            f"normalize_data = {self.normalize_data}\n"
            f"(model) input_size = {self.input_size}\n"
            f"(model) output_size = {self.output_size}\n"
            "\n\nTESTING\n"
            f"test_split_each_update = {self.test_split_each_update}\n"
            #f"test_subj_IDs = {self.test_subj_IDs}\n"  # This might not be behaving correctly, it isn't used with kfcv anyways...
            f"test_split_fraction = {self.test_split_fraction}\n"
            f"use_kfold_crossval = {self.use_kfold_crossval}\n"
            f"num_kfolds = {self.num_kfolds}\n")

        # current_fold=1 since fold is incremented BEFORE training/saving is completed
        if (self.use_kfold_crossval==False) or (self.use_kfold_crossval==True and self.current_fold==1):
            with open(self.paramtxt_file_path, 'w') as file:
                file.write(param_log_str)
                file.write(sequential_base_str)
                file.write(federated_base_str)
                file.write(cphs_base_str)
                if self.ewc_bool:
                    continual_base_str = ("\n\nCONTINUAL LEARNING PARAMS\n"
                    f"ewc_bool = {self.ewc_bool}\n"
                    f"fisher_multiplier = {self.fisher_mult}\n")
                    file.write(continual_base_str)
                if self.deep_bool:
                    deep_base_str = ("\n\nDEEP NETWORK HYPERPARAMETERS\n"
                        f"hidden_size = {self.hidden_size}\n"
                        f"sequence_length = {self.sequence_length}\n")
                    file.write(deep_base_str)
                if self.algorithm.upper()=="PERAVG" or self.algorithm=="PERFEDAVG" or self.algorithm=="PFA" or self.algorithm=="PFEDME":
                    perfedavg_param_str = ("\n\nPERFEDAVG\PFEDME PARAMS\n"
                        f"beta = {self.beta}\n")
                    file.write(perfedavg_param_str)

        if (personalized==True and ((len(self.rs_test_loss_per))!=0)) or (personalized==False and ((len(self.rs_test_loss))!=0)):
            print("File path: " + self.h5_file_path)
            #for client in self.clients:
            #    client.results_file_path = self.trial_result_path
            #    client.h5_file_path = self.h5_file_path

            with h5py.File(self.h5_file_path, 'w') as hf:
                if personalized:
                    hf.create_dataset('rs_test_loss_per', data=self.rs_test_loss_per)
                    hf.create_dataset('rs_train_loss_per', data=self.rs_train_loss_per)
                else:
                    hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                if self.sequential:
                    hf.create_dataset('curr_live_rs_test_loss', data=self.curr_live_rs_test_loss)
                    hf.create_dataset('prev_live_rs_test_loss', data=self.prev_live_rs_test_loss)
                    hf.create_dataset('unseen_live_rs_test_loss', data=self.unseen_live_rs_test_loss)

                # Save cross-cli test log if necessary (SERVERLOCAL ONLY AS OF 11/26/23):
                if self.test_against_all_other_clients:
                    hf.create_dataset('cross_client_loss_array', data=self.clii_on_clij_loss)
                    hf.create_dataset('cross_client_numsamples_array', data=self.clii_on_clij_numsamples)

                if self.save_client_loss_logs:
                    # Uhhh how is client_testing_log different from rs_test_loss again? ... 
                    ## rs_test_loss is the version averaged across clients?... I'm assuming client_testing_log doesnt include the round number...
                    ### Or is it called every round when each client gets evaluated? .......
                    group = hf.create_group('client_testing_logs')
                    for c in self.clients:
                        dataset_name = c.ID
                        data = c.client_testing_log  # Replace this with your actual data
                        group.create_dataset(dataset_name, data=data)

                if save_cost_func_comps:
                    #print(f'cost_func_comps_log: \n {self.cost_func_comps_log}\n')                   
                    G1 = hf.create_group('cost_func_tuples_by_client')
                    for idx, cost_func_comps in enumerate(self.cost_func_comps_log):
                        name_index = idx // len(self.condition_number_lst)
                        # TODO: should this be all_subj_IDs or its train equivalent? I think the train equivalent...
                        ## TODO: Need to update all the saving regardless... am I gonna save everything from each trial/fold? Makes the most sense / easiest
                        ## Could maybe do this by just adding which fold it is to the save string which should be fine...
                        ### TODO: Add fold to save string
                        # TODO: Check this
                        ## Actually I think this is completely broken...
                        ## Made more complicated by the extra conditions (which I am not using AFAIK)
                        if name_index >= len(self.all_subj_IDs):
                            # TODO: Check this
                            name_index = len(self.all_subj_IDs) - 1  # Ensure it doesn't exceed the last index
                        # TODO: Check this
                        name_str = self.all_subj_IDs[name_index] + "_C" + str(self.condition_number_lst[idx%len(self.condition_number_lst)])
                        G1.create_dataset(name_str, data=cost_func_comps)

                if save_gradient:
                    #print(f'gradient_norm_log: \n {self.gradient_norm_log}\n')
                    G2 = hf.create_group('gradient_norm_lists_by_client')
                    for idx, grad_norm_list in enumerate(self.gradient_norm_log):
                        name_index = idx // len(self.condition_number_lst)
                        # TODO: should this be all_subj_IDs or its train equivalent? I think the train equivalent...
                        ## TODO: Need to update all the saving regardless... am I gonna save everything from each trial/fold? Makes the most sense / easiest
                        ## Could maybe do this by just adding which fold it is to the save string which should be fine...
                        ### TODO: Add fold to save string
                        # TODO: Check this
                        if name_index >= len(self.all_subj_IDs):
                            # TODO: Check this
                            name_index = len(self.all_subj_IDs) - 1  # Ensure it doesn't exceed the last index
                        # TODO: Check this
                        name_str = self.all_subj_IDs[name_index] + "_C" + str(self.condition_number_lst[idx%len(self.condition_number_lst)])
                        G2.create_dataset(name_str, data=grad_norm_list)
        else:
            print("Saving failed.")


    #def save_fold_results(self):
    #    # TODO: Use this? I think the above save function can just be repurposed instead...
    #    raise ValueError("save_fold_results NEVER FINISHED")
    #
    #    algo = self.algorithm + "_" + self.goal # + "_" + str(self.times) 
    #
    #    if not os.path.exists(self.result_path):
    #        os.makedirs(self.result_path)
    #
    #    group = hf.create_group('client_testing_logs')
    #    for c in self.clients:
    #        dataset_name = c.ID
    #        data = c.client_testing_log  # Replace this with your actual data
    #        group.create_dataset(dataset_name, data=data)


    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            print(f"SB save_item() made directory! {self.save_folder_name}")
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def test_metrics(self):

        tsmt_start = time.time()

        ###############################
        # Idk if this part works... should get subsumed (and ideally not used) by Seq anyways...
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        ###############################
        
        num_samples = []
        tot_loss = []
        IDs = []
        if self.sequential:
            curr_live_loss = []
            curr_live_num_samples = []
            curr_live_IDs = []

            prev_live_loss = []
            prev_live_num_samples = []
            prev_live_IDs = []

            unseen_live_loss = []
            unseen_live_num_samples = []
            unseen_live_IDs = []

        for i, cID in enumerate(self.train_subj_IDs):
            # Get the actual client objects from the ID
            c = self.dict_map_subjID_to_clientobj[cID]

            #print(f"TEST_METRICS, USER {c.ID}")
            #if (self.sequential and c.ID in self.static_client_IDs):
            if self.sequential:
                # Test the global model not the clients' unchanging static model
                ## Gonna just leave it as testing the global model... so it is fair between all participants...
                ## Maybe make another one for testing local finetuned models...
                ## Need this change so it tests on prev live clients
                tl, ns = c.test_metrics(model_obj=self.global_model)
            else:
                tl, ns = c.test_metrics()
                #print(f"Client{i} test loss: {tl}, ns: {ns}, client bs: {c.batch_size}")
                # ^ ns is the cumulative number of seen samples NOT the batch_size

            if (not self.sequential) or (self.sequential and c.ID in self.static_client_IDs):
                # This is the ordinary nonseq sim case
                tot_loss.append(tl*1.0)
                num_samples.append(ns)
                IDs.append(c.ID)
            elif self.sequential and c.ID in [lc.ID for lc in self.live_clients]:
                # If it is the currently live client:
                ## I want to see its loss improving at the very least
                curr_live_loss.append(tl*1.0)
                curr_live_num_samples.append(ns)
                curr_live_IDs.append(c.ID)
            elif self.sequential and c.ID in self.prev_live_client_IDs:
                # If it is a previously live client:
                ## Intuitievly loss will go up but not too much hopefully
                ## Don't want to erase the gains on prev clients from learning on new clients
                prev_live_loss.append(tl*1.0)
                prev_live_num_samples.append(ns)
                prev_live_IDs.append(c.ID)
            elif self.sequential and c.ID in self.unseen_live_client_IDs:
                unseen_live_loss.append(tl*1.0)
                unseen_live_num_samples.append(ns)
                unseen_live_IDs.append(c.ID)
        if self.sequential:
            seq_metrics = [curr_live_loss, curr_live_num_samples, curr_live_IDs, prev_live_loss, prev_live_num_samples, prev_live_IDs, unseen_live_loss, unseen_live_num_samples, unseen_live_IDs]
        else:
            seq_metrics = None

        print(f"Server test metrics finished in {time.time() - tsmt_start}")

        # This also ought to be flipped around...
        return IDs, num_samples, tot_loss, seq_metrics


    def train_metrics(self):
        '''
        Never added unseen training loss to outputs returned... don't really care right now.
        Presumably the train and test on the unseen clients is the same?
        Really ought to switch to testing on entirely new clients...
        This is a problem that is unaddressed with test_metrics() as well...
        '''
        
        trmt_start = time.time()

        if self.eval_new_clients and self.num_new_clients > 0:
            # if eval new client AND we actually have new clients, then return this
            # maps to depreceated values I think...
            print("KAI: Returned early for some reason, idk what this code is doing")
            return [0], [1], [0]

        num_samples = []
        tot_loss = []
        IDs = []
        if self.sequential:
            curr_live_loss = []
            curr_live_num_samples = []
            curr_live_IDs = []

            prev_live_loss = []
            prev_live_num_samples = []
            prev_live_IDs = []
        # ITERATES OVER ALL CLIENTS, THEN SUMS ALL CLIENT LOSSES!
        for i, cID in enumerate(self.train_subj_IDs):
            # Get the actual client objects from the ID
            c = self.dict_map_subjID_to_clientobj[cID]
            #print(f"TRAIN_METRICS, USER {c.ID}")
            if (self.sequential and c.ID in self.static_client_IDs):
                # Test the current global model on the training data of the static clients!
                ## The global model should be performing better as time goes on, even tho it hasn't seen these clients?
                ## If fresh model, then it has never seen these clients, else it could be pretrained on them and we are looking for retention...
                tl, ns = c.train_metrics(model_obj=self.global_model)
            else:
                # If you're anyone else, you test your current local model on your training data
                tl, ns = c.train_metrics()
            
            # DETERMINE WHERE TO SAVE THE PERFORMANCE (WHICH IS THE APPROPRIATE LOG)
            if (not self.sequential) or (self.sequential and c.ID in self.static_client_IDs):
                # This is the ordinary nonseq sim case, and also for static seq clients
                tot_loss.append(tl*1.0)
                num_samples.append(ns)
                IDs.append(c.ID)
            elif self.sequential and c.ID in [lc.ID for lc in self.live_clients]:
                # If it is the currently live client:
                ## This loss should be improving
                curr_live_loss.append(tl*1.0)
                curr_live_num_samples.append(ns)
                curr_live_IDs.append(c.ID)
            elif self.sequential and c.ID in self.prev_live_client_IDs:
                # If it is a previously live client:
                ## Intuitievly loss will go up but not too much hopefully
                ## Don't want to erase the gains on prev clients from learning on new clients
                prev_live_loss.append(tl*1.0)
                prev_live_num_samples.append(ns)
                prev_live_IDs.append(c.ID)
            elif self.sequential and c.ID in self.live_client_IDs_queue:
                # Eg it hasn't been trained/called yet
                pass
            elif self.sequential:
                raise ValueError("This isn't supposed to run...")
            else:
                raise ValueError("This isn't supposed to run...")
        #IDs = [c.ID for c in self.clients]
        if self.sequential:
            # Why doesn't this one do unseen testing? I guess unseen implies testing not training
            seq_metrics = [curr_live_loss, curr_live_num_samples, curr_live_IDs, prev_live_loss, prev_live_num_samples, prev_live_IDs]
        else:
            seq_metrics = None

        print(f"Server train metrics finished in {time.time() - trmt_start}")

        # This really ought to be flipped to loss, ns, IDs ...
        return IDs, num_samples, tot_loss, seq_metrics

    # evaluate selected clients
    def evaluate(self, train=True, test=True):
        '''
        KAI Docstring
        This func runs test_metrics and train_metrics, and then sums all of ...
        Previously, test_metrics and train_metrics were collecting the losses on ALL clients (even the untrained ones...)
        I switched that (5/31/23) to be just the selected clients, the idea being that ALL clients explode the loss func
        ^ 1/12/24 uhh what? No idea where this change is. Idk if I really want that... prob shold stay with what they had
        ^^ Unless that was related to classification accuracy and thus I couldn't use it
        ^^^ I don't even see self.clients...

        This is kind of annoying, it's basically a logging function that wraps the test_metrics and train_metrics functions...
        IMO this functionality would be better used inside of those functions... I'm not gonna change it now tho (1/12/24)
        '''
        if self.verbose:
            print("Serverbase evaluate()")
        if test:
            stats = self.test_metrics()
            if self.verbose:
                print(f"Len of test_metrics() output: {len(stats[0])}")
                print(f"Sum (ns) of test_metrics() output: {sum(stats[1])}")
            #test_loss = sum(stats[2])*1.0 / len(stats[2])  # Idk what this was doing either. Not relevant to us...
            #test_loss = sum(stats[2])*1.0  # Used to return test_acc, test_num, auc; idk what it is summing tho (or why auc wouldn't be a scalar...)
            test_loss = stats[2]
            test_samples_per_round = stats[1]

            avg_test_loss = sum(test_loss)/sum(test_samples_per_round)
            self.rs_test_loss.append(avg_test_loss)

            if self.sequential:
                # seq_stats <-- [curr_live_loss, curr_live_num_samples, curr_live_IDs, prev_live_loss, prev_live_num_samples, prev_live_IDs, unseen_live_loss, unseen_live_num_samples, unseen_live_IDs]
                # Hmm do I need to save/use the actual IDs at all? Do I care? Don't think so...
                seq_stats = stats[3]
                if len(seq_stats[0])!=0:
                    #self.curr_live_rs_test_loss.append(sum(seq_stats[0])/len(seq_stats[0]))
                    self.curr_live_rs_test_loss.append(sum(seq_stats[0])/sum(seq_stats[1]))
                if len(seq_stats[3])!=0:
                    #self.prev_live_rs_test_loss.append(sum(seq_stats[3])/len(seq_stats[3]))
                    self.prev_live_rs_test_loss.append(sum(seq_stats[3])/sum(seq_stats[4]))
                if len(seq_stats[6])!=0:
                    #self.unseen_live_rs_test_loss.append(sum(seq_stats[6])/len(seq_stats[6]))
                    self.unseen_live_rs_test_loss.append(sum(seq_stats[6])/sum(seq_stats[7]))

            #assert(test_loss<1e5)
            print("Averaged Test Loss: {:.5f}".format(avg_test_loss))

        if train:
            stats_train = self.train_metrics()
            if self.verbose:
                print(f"Len of train_metrics() output: {len(stats_train[0])}")
                print(f"Sum (ns) of train_metrics() output: {len(stats_train[1])}")
            avg_train_loss = sum(stats_train[2]) / sum(stats_train[1])
            self.rs_train_loss.append(avg_train_loss)
            # I'm not even recording the training seq metrics right now... don't really care
            # Do I even have those lol

            print("Averaged Train Loss: {:.5f}".format(avg_train_loss))

            # If the average loss is unreasonably high just abort the run
            if avg_train_loss>self.loss_threshold:
                # Log training loss up to this point...
                self.save_results(personalized=self.personalized_algo_bool, save_cost_func_comps=True, save_gradient=True)
                raise ValueError('Averaged training loss exceeded the maximum loss threshold, aborting training.')


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        print("Running check_done")
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


    def set_new_clients(self, clientObj):
        if self.num_new_clients==0:
            pass
        else:
            base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Data\\CPHS_EMG\\Subject_Specific_Files\\'
            for i in range(self.num_clients, self.num_clients + self.num_new_clients):
                # Idk I guess I can keep the condition iter? Idk why I would want to turn it off other than not expecting it
                for j in self.condition_number_lst:
                    print(f"SB Set New Client: iter iter {i}, cond number: {str(j)}")
                    ID_str = self.all_subj_IDs[i]
                    client = clientObj(self.args, 
                                        ID=ID_str, 
                                        samples_path = base_data_path + 'S' + ID_str[-3:] + "_TrainData_8by20770by64.npy", 
                                        labels_path = base_data_path + 'S' + ID_str[-3:] + "_Labels_8by20770by2.npy", 
                                        condition_number = j-1, 
                                        train_slow=False, 
                                        send_slow=False)
                    self.clients.append(client)


    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        print("fine_tuning_new_clients USES GLOBAL MODEL!!!")
        for client in self.new_clients_obj_lst:
            client.set_parameters(self.global_model)
            for _ in range(self.fine_tuning_epoch):
                client.train()


    # evaluating on new clients
    def test_metrics_new_clients(self):
        # UPDATE THIS FOR SEQUENTIAL????
        # What's the diff even between this and my sequential?
        ## My seq tests on live and prev clients, this is testing on entirely new
        num_samples = []
        tot_loss = []
        for c in self.clients:
            tl, ns = c.test_metrics()
            tot_loss.append(tl*1.0)
            num_samples.append(ns)

        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, tot_loss

    def plot_results(self, plot_this_list_of_vecs=None, list_of_labels=None, plot_train=True, plot_test=True, plot_seq=True, my_title=None):
        # Apparently this hasn't run yet? ...
        ## kfold_suffix must be updated since there is a new fold idx, which affects the rest of these file names...
        self.kfold_suffix = f"_kfold{self.current_fold}" if self.use_kfold_crossval else ""
        self.trial_result_path = self.script_directory+self.relative_path + self.kfold_suffix
        if not os.path.exists(self.trial_result_path):
            os.makedirs(self.trial_result_path)
        
        if plot_this_list_of_vecs is not None and list_of_labels is not None:
            for vec, vec_label in zip(plot_this_list_of_vecs, list_of_labels):
                plt.plot(range(len(vec)), vec, label=vec_label)
        else:
            if self.algorithm.upper() == 'LOCAL':
                #self.clii_on_clij_loss --> 14x14xR (R is number of global rounds)
                # I need to reduce this matrix down to 1x1x(r in R % ccm == 0)
                # First, we only need the diagonal rows, as this is where the averages should be saved
                avg_cc_nestedlst = [[] for _ in range(len(self.clients))]
                for i in range(len(self.clients)):
                    for j in range(self.global_rounds):
                        if j%self.cross_client_modulus==0:
                            # ... why are all avg_cc_nestedlst entries the same......
                            avg_cc_nestedlst[i].append(self.clii_on_clij_loss[i, i, j])
                # Now, we need to extract every rth value, where r is set directly by ccm
                ## Should be easy to do by filter...
                # Make an x_axis determined by ccm:
                spaced_xs = list(range(0, self.global_rounds, self.cross_client_modulus))
                local_x_axis = [int(ele) for ele in spaced_xs]
                for i in range(len(self.clients)):
                    # Fix this label... switch to subjectID
                    plt.plot(local_x_axis, avg_cc_nestedlst[i], label=self.clients[i].ID)
            else:
                if plot_test:
                    plt.plot(range(len(self.rs_test_loss)), self.rs_test_loss, label='Test')
                if plot_train:
                    plt.plot(range(len(self.rs_train_loss)), self.rs_train_loss, label='Train')
            if plot_seq==True and self.sequential==True:
                # cl should be the same length
                ## Yah I have no idea why I need/have an offset here, it should always be being written to...
                cl_offset_diff = len(self.rs_test_loss) - len(self.curr_live_rs_test_loss)
                cl_x_axis = np.array(range(len(self.curr_live_rs_test_loss))) + cl_offset_diff

                # pl should start late tho, I believe
                pl_offset_diff = len(self.rs_test_loss) - len(self.prev_live_rs_test_loss)
                pl_x_axis = np.array(range(len(self.prev_live_rs_test_loss))) + pl_offset_diff

                # ul should be shorter but it just isn't written to at the end I think? 
                ## So theoretically I shoudln't have to do anything at all here...
                ul_x_axis = np.array(range(len(self.unseen_live_rs_test_loss)))

                plt.plot(cl_x_axis, self.curr_live_rs_test_loss, label='Current Testing')
                plt.plot(pl_x_axis, self.prev_live_rs_test_loss, label='Previous Testing')#, width=5, alpha=0.5)
                plt.plot(ul_x_axis, self.unseen_live_rs_test_loss, label='Unseen Testing')

        if my_title is None:
            plt.title("Train/test loss")
        else:
            plt.title(my_title)
        plt.xlabel("Iteration Number")
        plt.ylabel("Loss")
        if self.test_against_all_other_clients:
            plt.legend(fontsize='small')
        else:
            plt.legend()
        plt.savefig(self.trial_result_path + '\\TrainTestLossCurves.png', format='png')
        plt.show()
        
    