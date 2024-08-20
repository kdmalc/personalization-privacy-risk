import numpy as np
import random
import copy
from datetime import datetime
import os
import h5py

from experiment_params import *
from cost_funcs import *
from fl_sim_base import *
        
        
class Server(ModelBase):
    def __init__(self, ID, D0, opt_method, global_method, all_clients, smoothbatch_lr=0.75, C=0.35, test_split_type='kfoldcv', 
                 num_kfolds=5, test_split_frac=0.3, current_round=0, PCA_comps=64, verbose=False, validate_memory_IDs=True, save_client_loss_logs=True, 
                 sequential=False):
        super().__init__(ID, D0, opt_method, smoothbatch_lr=smoothbatch_lr, current_round=current_round, PCA_comps=PCA_comps, 
                         verbose=verbose, num_clients=14, log_init=0)
        
        self.type = 'Server'
        self.save_client_loss_logs = save_client_loss_logs
        self.sequential = sequential

        # CLIENT LISTS!
        self.num_avail_clients = 0
        self.available_clients_lst = [0]*len(all_clients)  # Why is this not just an empty list...
        # ^ This should really be: [cli for cli in all_clients if cli.availability==1]
        self.num_chosen_clients = 0
        self.chosen_clients_lst = [0]*len(all_clients)  # Why is this not just an empty list...
        self.all_clients = all_clients  # ALL TRAIN AND VAL AND STATIC/UNAVAILABLE/QUEUED CLIENTS ARE PASSED IN HERE!!!
        self.num_total_clients = len(self.all_clients)
        # THE BELOW SHOULD NOT CHANGE DURING THE RUN!
        ## Maybe train_clients can/should change, to reflect availability in the seq case? ...
        self.train_clients = [cli for cli in self.all_clients if cli.val_set is False]
        self.val_clients = [cli for cli in self.all_clients if cli.val_set is True]
        self.num_train_clients = len(self.train_clients)
        self.num_test_clients = len(self.val_clients)
        # SET AVAILABLE CLIENT LIST
        self.set_available_clients_list()
        # Init all train clients with simulate_streaming so they have self.s, self.F, and self.p_reference
        for client in self.train_clients:
            client.simulate_data_stream()

        self.global_method = global_method.upper()
        print(f"Running the {self.global_method} algorithm as the global method!")
        self.C = C  # Fraction of clients to use each round

        # TESTING
        self.test_split_type = test_split_type.upper()
        self.num_kfolds = num_kfolds
        self.test_split_frac = test_split_frac

        # SAVE FILE RELATED
        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        self.str_current_datetime = str(current_datetime)
        # Get the directory of the current script
        self.script_directory = os.path.dirname(os.path.abspath(__file__))  # This returns the path to serverbase... so don't index the end of the path
        # Relative path to results dir
        self.result_path = "\\results\\"
        # Specify the relative path from the script's directory
        self.relative_path = self.result_path + self.str_current_datetime + "_" + self.global_method
        # Combine the script's directory and the relative path to get the full path
        self.trial_result_path = self.script_directory + self.relative_path
        self.h5_file_path = os.path.join(self.trial_result_path, f"{self.opt_method}_{self.global_method}")
        self.paramtxt_file_path = os.path.join(self.trial_result_path, "param_log.txt")
        if not os.path.exists(self.trial_result_path):
            os.makedirs(self.trial_result_path)
        
                
    # Main Loop
    def execute_FL_loop(self):
        # Update global round number
        self.current_round += 1
        
        if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
            # Choose fraction C of available clients
            self.set_available_clients_list()
            self.choose_clients()
            for my_client in self.all_clients:  
                if my_client.val_set==True:
                    # This is a val client, so don't log anything
                    # Could be included in train_metrics but doesnt need to be
                    continue
                elif my_client.availability==False:
                    # This is a queue'd client or something
                    ## Should it carry forward their loss...
                    ## I think carrying forward is kind of stupid...
                    ## Is this supposed to carry a value ()
                    ## Nonavail should still be train_metrics()'d over I think..., just not trained
                    raise ValueError("Sequential case not implemented yet")
                elif my_client not in self.chosen_clients_lst:
                    # Just not getting trained this round
                    continue
                # THESE ARE THE CLIENTS WHICH ACTUALLY GET TRAINED!
                my_client.latest_global_round = self.current_round          
                # Send those clients the current global model
                my_client.global_w = copy.deepcopy(self.w)
                my_client.execute_training_loop()
            # AGGREGATION
            self.agg_local_weights()  # This func sets self.w, eg the new decoder
            # GLOBAL SmoothBatch
            self.w = self.smoothbatch_lr*self.w_prev + ((1 - self.smoothbatch_lr)*self.w)
        elif self.global_method=='NOFL':
            # TODO: Is NoFL just supposed to be the Local CPHS sims... if so this is fine I think 
            for my_client in self.all_clients:  
                if my_client.val_set==True:
                    # This is a val client, so don't log anything
                    # Could be included in train_metrics but doesnt need to be
                    continue
                elif my_client.availability==False:
                    # This is a queue'd client or something
                    ## Should it carry forward their loss...
                    ## I think carrying forward is kind of stupid...
                    ## Is this supposed to carry a value ()
                    ## Nonavail should still be train_metrics()'d over I think..., just not trained
                    raise ValueError("Sequential case not implemented yet")
                # THESE ARE THE CLIENTS WHICH ACTUALLY GET TRAINED!
                my_client.latest_global_round = self.current_round          
                my_client.execute_training_loop()
        else:
            raise('Method not currently supported, please reset method to FedAvg')
        
        # Save the new decoder to the log
        self.dec_log.append(copy.deepcopy(self.w))
        # Run train_metrics and test_metrics to log performance on training/testing data
        for client_idx, my_client in enumerate(self.train_clients): # Eg all train-able clients (no witheld val clients from kfoldcv)
            # Reset all clients so no one is chosen for the next round
            my_client.chosen_status = 0
            # test_metrics for all clients
            if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
                global_test_loss, _ = my_client.test_metrics(self.w, 'global')
                local_test_loss, _ = my_client.test_metrics(my_client.w, 'local')

                global_train_loss, _ = my_client.train_metrics(self.w, 'global')
                local_train_loss, _ = my_client.train_metrics(my_client.w, 'local')
            elif self.global_method=='NOFL':
                global_test_loss = 0
                local_test_loss, _ = my_client.train_metrics(my_client.w, 'local') 

                global_train_loss = 0
                local_train_loss, _ = my_client.train_metrics(my_client.w, 'local') 
            
            if client_idx!=0:
                running_global_test_loss += global_test_loss
                running_local_test_loss += local_test_loss

                running_global_train_loss += global_train_loss
                running_local_train_loss += local_train_loss
            else:
                running_global_test_loss = global_test_loss
                running_local_test_loss = local_test_loss

                running_global_train_loss = global_train_loss
                running_local_train_loss = local_train_loss

        # SERVER AVERAGE PERFORMANCE
        # Divide by the number of clients to get average loss per client
        ## For local, having the individual client logs would be better but they would probably get averaged anyways when plotting so
        self.local_test_error_log.append(running_local_test_loss / len(self.train_clients))
        self.local_train_error_log.append(running_local_train_loss / len(self.train_clients))
        if self.global_method!='NOFL':
            self.global_test_error_log.append(running_global_test_loss / len(self.train_clients))
            self.global_train_error_log.append(running_global_train_loss / len(self.train_clients))
            
            
    def set_available_clients_list(self):
        # TODO come back to this depending on how available_clients_full_idx_lst shakes out...
        self.available_clients_full_idx_lst = [0]*len(self.train_clients)
        for idx, my_client in enumerate(self.train_clients):
            if my_client.availability:
                self.available_clients_full_idx_lst[idx] = my_client
        # cli can be 0 if that client is not available this round... --> I think self.train_clients would need to be self.all_clients above tho...
        self.available_clients_lst = [cli for cli in self.available_clients_full_idx_lst if cli != 0]
        self.num_avail_clients = len(self.available_clients_lst)
    

    def choose_clients(self):
        # Check what client are available this round
        self.set_available_clients_list()
        # Now choose frac C clients from the resulting available clients
        if self.num_avail_clients > 0:
            self.num_chosen_clients = int(np.ceil(self.num_avail_clients*self.C))
            if self.num_chosen_clients<1:
                raise ValueError(f"ERROR: Chose {self.num_chosen_clients} clients for some reason, must choose more than 1")
            # Right now it chooses 2 at random: 14*.1=1.4 --> 2
            self.chosen_clients_lst = random.sample(self.available_clients_lst, len(self.available_clients_lst))[:self.num_chosen_clients]
            for my_client in self.chosen_clients_lst:
                my_client.chosen_status = 1
        else:
            raise(f"ERROR: Number of available clients must be greater than 0: {self.num_avail_clients}")

    
    def agg_local_weights(self):
        self.w_prev = copy.deepcopy(self.w)

        # From McMahan 2017 (vanilla FL)
        summed_num_datapoints = 0
        for my_client in self.chosen_clients_lst:
            summed_num_datapoints += my_client.learning_batch
        # Aggregate local model weights, weighted by normalized local learning rate
        ## Learning rate or number of datapoints? I believe it is number of datapoints, as implemented
        aggr_w = np.zeros((2, self.PCA_comps))
        for my_client in self.chosen_clients_lst:
            aggr_w += (my_client.learning_batch/summed_num_datapoints) * my_client.w
        
        self.w = aggr_w


    def save_header(self):

        sequential_base_list = [
            "\n\nSEQUENTIAL\n",
            f"sequential = {self.sequential}\n"]
        # TODO: This has yet to be implemented
        if self.sequential:
            sequential_base_list.extend([
                f"live_client_IDs_queue = {self.live_client_IDs_queue}\n",
                f"static_client_IDs = {self.static_client_IDs}\n",
                f"num_liveseq_rounds_per_seqclient = {self.num_liveseq_rounds_per_seqclient}\n",
                f"prev_model_directory = {self.prev_model_directory}"])
        sequential_base_str = ''.join(sequential_base_list)
        
        federated_base_str = ("\n\nFEDERATED LEARNING PARAMS\n"
            f"local_round_threshold = {self.train_clients[0].local_round_threshold}\n")

        cphs_base_str = ("\n\nSIMULATION PARAMS\n"
            f"starting_update = {self.starting_update}\n"
            #f"all_subj_IDs = {self.all_subj_IDs}\n"
            #f"condition_number_lst = {self.condition_number_lst}\n"
            f"total effective clients = {self.num_train_clients}\n"
            f"smoothbatch_lr = {self.smoothbatch_lr}\n")

        param_log_str = (
            "BASE\n"
            f"algorithm = {self.global_method}\n"
            #f"model = {self.model}\n"  # TODO: Make this return/print a string...
            "\n\nMODEL HYPERPARAMETERS\n"
            f"lambdaF = {self.alphaF}\n"
            f"lambdaD = {self.alphaD}\n"
            f"lambdaE = {self.alphaE}\n"
            f"global_rounds = {self.global_rounds}\n"
            #f"local_epochs = {self.local_epochs}\n"
            #f"batch_size = {self.batch_size}\n"
            #f"local_learning_rate = {self.local_learning_rate}\n"
            #f"learning_rate_decay = {self.learning_rate_decay}\n"
            #f"learning_rate_decay_gamma = {self.learning_rate_decay_gamma}\n"
            f"optimizer = {self.opt_method}\n"
            f"pca_channels = {self.PCA_comps}\n"
            #f"(model) input_size = {self.input_size}\n"
            #f"(model) output_size = {self.output_size}\n"
            "\n\nTESTING\n"
            f"test_split_fraction = {self.test_split_frac}\n"
            f"num_kfolds = {self.num_kfolds}\n")

        with open(self.paramtxt_file_path, 'w') as file:
            file.write(param_log_str)
            file.write(sequential_base_str)
            file.write(federated_base_str)
            file.write(cphs_base_str)
            #if self.ewc_bool:
            #    continual_base_str = ("\n\nCONTINUAL LEARNING PARAMS\n"
            #    f"ewc_bool = {self.ewc_bool}\n"
            #    f"fisher_multiplier = {self.fisher_mult}\n")
            #    file.write(continual_base_str)
            #if self.deep_bool:
            #    deep_base_str = ("\n\nDEEP NETWORK HYPERPARAMETERS\n"
            #        f"hidden_size = {self.hidden_size}\n"
            #        f"sequence_length = {self.sequence_length}\n")
            #    file.write(deep_base_str)
            if "PFA" in self.global_method:
                perfedavg_param_str = ("\n\nPERFEDAVG PARAMS\n"
                    f"beta = {self.all_clients[0].beta}\n")
                file.write(perfedavg_param_str)


    def save_results_h5(self, save_cost_func_comps=False, save_gradient=False):

        if len(self.local_test_error_log)!=0:
            print("File path: " + self.h5_file_path + ".h5")

            if self.test_split_type=="KFOLDCV":
                save_filename = self.h5_file_path + f"_KFold{self.current_fold}.h5"
            else:
                save_filename = self.h5_file_path + ".h5"

            with h5py.File(save_filename, 'w') as hf:
                if self.global_method!="NOFL":
                    hf.create_dataset('global_test_error_log', data=self.global_test_error_log)
                    hf.create_dataset('global_train_error_log', data=self.global_train_error_log)
                else:
                    # TODO: Shouldn't this get saved always...
                    hf.create_dataset('local_test_error_log', data=self.local_test_error_log)
                    hf.create_dataset('local_train_error_log', data=self.local_train_error_log)

                # TODO: This feature got removed... 
                if self.sequential:
                    hf.create_dataset('curr_live_rs_test_loss', data=self.curr_live_rs_test_loss)
                    hf.create_dataset('prev_live_rs_test_loss', data=self.prev_live_rs_test_loss)
                    hf.create_dataset('unseen_live_rs_test_loss', data=self.unseen_live_rs_test_loss)

                if self.save_client_loss_logs:
                    group = hf.create_group('client_testing_logs')
                    for c in self.all_clients:
                        dataset_name = c.ID
                        data = c.client_testing_log  # Replace this with your actual data
                        group.create_dataset(dataset_name, data=data)

                if save_cost_func_comps:
                    # TODO: This is probably not implemented the same
                    G1 = hf.create_group('cost_func_tuples_by_client')
                    for idx, cost_func_comps in enumerate(self.cost_func_comps_log):
                        name_index = idx // len(self.condition_number_lst)
                        if name_index >= len(self.all_subj_IDs):
                            name_index = len(self.all_subj_IDs) - 1  # Ensure it doesn't exceed the last index
                        name_str = self.all_subj_IDs[name_index] + "_C" + str(self.condition_number_lst[idx % len(self.condition_number_lst)])
                        G1.create_dataset(name_str, data=cost_func_comps)

                if save_gradient:
                    # TODO: As of now, I don't think gradients are tracked...
                    G2 = hf.create_group('gradient_norm_lists_by_client')
                    for idx, grad_norm_list in enumerate(self.gradient_norm_log):
                        name_index = idx // len(self.condition_number_lst)
                        if name_index >= len(self.all_subj_IDs):
                            name_index = len(self.all_subj_IDs) - 1  # Ensure it doesn't exceed the last index
                        name_str = self.all_subj_IDs[name_index] + "_C" + str(self.condition_number_lst[idx % len(self.condition_number_lst)])
                        G2.create_dataset(name_str, data=grad_norm_list)
        else:
            print("Saving failed.")

        
