import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import copy
from matplotlib import pyplot as plt

from experiment_params import *
from cost_funcs import *
import time
import pickle
from sklearn.decomposition import PCA

from fl_sim_base import *
        
        
class Server(ModelBase):
    def __init__(self, ID, D0, method, all_clients, smoothbatch=1, C=0.35, normalize_dec=True, test_split_type='end', use_up16_for_test=True, test_split_frac=0.3, current_round=0, PCA_comps=10, verbose=False, APFL_Tau=10, copy_type='deep', validate_memory_IDs=True):
        super().__init__(ID, D0, method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, verbose=verbose, num_participants=14, log_init=0)
        self.type = 'Server'
        self.num_avail_clients = 0
        self.available_clients_lst = [0]*len(all_clients)
        self.num_chosen_clients = 0
        self.chosen_clients_lst = [0]*len(all_clients)
        self.all_clients = all_clients
        self.C = C  # Fraction of clients to use each round
        self.experimental_inclusion_round = [0]
        self.init_lst = [self.log_init]*self.num_participants
        self.normalize_dec = normalize_dec
        self.set_available_clients_list(init=True)
        self.validate_memory_IDs = validate_memory_IDs
        self.copy_type = copy_type
        if self.method=='APFL':
            self.set_available_clients_list()
            self.choose_clients()
            self.K = len(self.chosen_clients_lst)
            self.tau = APFL_Tau
        self.test_split_type = test_split_type
        self.test_split_frac = test_split_frac
        self.use_up16_for_test = use_up16_for_test

                
    # 0: Main Loop
    def execute_FL_loop(self):
        # Update global round number
        self.current_round += 1
        
        if 'FedAvg' in self.method:  # OR EQUIVALENTLY: if self.method in ['FedAvg', 'FedAvgSB']:
            # Choose fraction C of available clients
            self.set_available_clients_list()
            self.choose_clients()
            # Let those clients train (this autoselects the chosen_client_lst to use)
            self.train_client_and_log(client_set=self.chosen_clients_lst)
            # AGGREGATION
            self.w_prev = copy.deepcopy(self.w)
            self.agg_local_weights()  # This func sets self.w, eg the new decoder
            # GLOBAL SmoothBatch
            #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
            #^ Note when self.smoothbatch=1 (default), we just keep the new self.w (no mixing)
            if self.method=='FedAvg':
                self.w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.w_prev)
                # Eg don't do a global-global smoothbatch for the other cases
        elif self.method=='NoFL':
            self.train_client_and_log(client_set=self.all_clients)
        elif self.method=='APFL':
            t = self.current_round
            # They wrote: "if t not devices Tau then" but that seem like it would only run 1 update per t, 
            # AKA 50 t's to select new clients.  I'll write it like they did ig...
            if t%self.tau!=0:
                self.train_client_and_log(client_set=self.chosen_clients_lst)
                for my_client in list(set(self.available_clients_lst) ^ set(self.chosen_clients_lst)):
                    # Otherwise indices will break when calculating finalized running terms
                    my_client.p.append(my_client.p[-1])
            else:
                # Carry forward the existing costs in the logs so the update/indices still match
                for my_client in self.all_clients:
                    # This should do no training but log everything we need I think?
                    self.train_client_and_log([])
                #    # Do I need to go in and edit the current round or can I just leave it? I think just leave it
                #    my_client.local_error_log.append(self.local_error_log[-1])
                #    my_client.global_error_log.append(self.global_error_log[-1])
                #    my_client.pers_error_log.append(self.pers_error_log[-1])
                
                # Aggregate global dec every tau iters
                running_global_dec = np.zeros((2, self.PCA_comps))
                for my_client in self.chosen_clients_lst:
                    running_global_dec += my_client.global_w
                self.w = (1/self.K)*running_global_dec
                self.set_available_clients_list()
                self.choose_clients()
                self.K = len(self.chosen_clients_lst)
                # Presumably len(self.chosen_clients_lst) will always be the same (same C)... 
                # otherwise would need to re-set K as so:
                #self.K = len(self.chosen_clients_lst)

                for my_client in self.chosen_clients_lst:
                    if self.copy_type == 'deep':
                        self.global_w = copy.deepcopy(self.w)
                    elif self.copy_type == 'shallow':
                        self.global_w = copy.copy(self.w)
                    elif self.copy_type == 'none':
                        self.global_w = self.w
                    else:
                        raise ValueError("copy_type must be set to either deep, shallow, or none")
        else:
            raise('Method not currently supported, please reset method to FedAvg')
        
        # Save the new decoder to the log
        self.dec_log.append(copy.deepcopy(self.w))
        # Run test_metrics to generate performance on testing data
        for client_idx, my_client in enumerate(self.all_clients):
            # Reset all clients so no one is chosen for the next round
            if type(my_client)==int:
                raise TypeError("All my clients are all integers...")
            my_client.chosen_status = 0
            # test_metrics for all clients
            if self.method=='FedAvg':
                global_test_loss, global_test_pred = my_client.test_metrics(self.w, 'global')
                local_test_loss, local_test_pred = my_client.test_metrics(my_client.w, 'local')
            elif self.method=='NoFL':
                global_test_loss = 0
                local_test_loss, local_test_pred = my_client.test_metrics(my_client.w, 'local') 
            #
            if client_idx!=0:
                running_global_test_loss += np.array(global_test_loss)
                running_local_test_loss += np.array(local_test_loss)
            else:
                running_global_test_loss = np.array(global_test_loss)
                running_local_test_loss = np.array(local_test_loss)
        if self.method=='FedAvg':
            self.global_test_error_log = running_global_test_loss / len(self.all_clients)
            self.local_test_error_log = running_local_test_loss / len(self.all_clients)
        elif self.method=='NoFL':
            self.local_test_error_log = running_local_test_loss / len(self.all_clients)
            
        
    # 1.1
    def set_available_clients_list(self, init=False):
        self.num_avail_clients = 0
        self.available_clients_lst = [0]*len(self.all_clients)
        for idx, my_client in enumerate(self.all_clients):
            if my_client.availability:
                self.available_clients_lst[idx] = my_client
                self.num_avail_clients += 1
                if init:
                    # Pass down the global METHOD (NOT THE WEIGHTS!!)
                    my_client.global_method = self.method
    
    # 1.2
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
            
    # 2
    def train_client_and_log(self, client_set):
        current_local_lst = []
        current_global_lst = []
        current_pers_lst = []
        for my_client in self.available_clients_lst:  # Implications of using this instead of all_clients?
            my_client.latest_global_round = self.current_round
            #^ Need to overwrite client with the curernt global round, for t later
            
            # This isn't great code because it checks the init every single time it runs
            #  Maybe move this to be before this loop?
            if len(self.local_error_log)==0:  #self.local_error_log[-1]==self.log_init:
                local_init_carry_val = 0
                global_init_carry_val = 0
                pers_init_carry_val = 0
            else:
                local_init_carry_val = self.local_error_log[-1][my_client.ID][2]#[0]
                if self.method != 'NoFL':
                    global_init_carry_val = self.global_error_log[-1][my_client.ID][2]#[0]
                if self.method in self.pers_methods:
                    pers_init_carry_val = self.pers_error_log[-1][my_client.ID][2]#[0]
                    
            if my_client in client_set:
                # Send those clients the current global model
                if self.copy_type == 'deep':
                    my_client.global_w = copy.deepcopy(self.w)
                elif self.copy_type == 'shallow':
                    my_client.global_w = copy.copy(self.w)
                elif self.copy_type == 'none':
                    my_client.global_w = self.w
                else:
                    raise ValueError("copy_type must be set to either deep, shallow, or none")
                
                my_client.execute_training_loop()
                
                if self.validate_memory_IDs:
                    assert(id(my_client.w)!=id(self.w))
                    assert(id(my_client.w)!=id(my_client.global_w))
                
                current_local_lst.append((my_client.ID, self.current_round, 
                                          my_client.eval_model(which='local')))
                if self.method != 'NoFL':
                    current_global_lst.append((my_client.ID, self.current_round, 
                                               my_client.eval_model(which='global')))
                if self.method in self.pers_methods:
                    current_pers_lst.append((my_client.ID, self.current_round, 
                                               my_client.eval_model(which='pers')))
            else:
                current_local_lst.append((my_client.ID, self.current_round, 
                                          local_init_carry_val))
                if self.method != 'NoFL':
                    current_global_lst.append((my_client.ID, self.current_round,
                                               global_init_carry_val))
                if self.method in self.pers_methods:
                    current_pers_lst.append((my_client.ID, self.current_round,
                                               pers_init_carry_val))
        # Append (ID, COST) to SERVER'S error log.  
        #  Note that round is implicit, it is just the index of the error log
        if self.local_error_log==self.init_lst:
            # Overwrite the [(0,0)] hold... why not just init to this then...
            self.local_error_log = []
            self.local_error_log.append(current_local_lst)
        else:
            self.local_error_log.append(current_local_lst)
        if self.method != 'NoFL':
            # NoFL case has no global model since there's... no FL
            if self.global_error_log==self.init_lst:
                self.global_error_log = []
                self.global_error_log.append(current_global_lst)
            else:
                self.global_error_log.append(current_global_lst)
        if self.method in self.pers_methods:
            if self.pers_error_log==self.init_lst:
                self.pers_error_log = []
                self.pers_error_log.append(current_pers_lst)
            else:
                self.pers_error_log.append(current_pers_lst)
    
    # 3
    def agg_local_weights(self):
        # From McMahan 2017 (vanilla FL)
        summed_num_datapoints = 0
        for my_client in self.chosen_clients_lst:
            summed_num_datapoints += my_client.learning_batch
        # Aggregate local model weights, weighted by normalized local learning rate
        aggr_w = np.zeros((2, self.PCA_comps))
        ###########################################################################################################################
        for my_client in self.chosen_clients_lst:
            # Hmmm should I actually be normalizing (or rather, scaling) the models before aggregation?
            # ^Breaks the cost func locally but globally I don't thikn it should matter?
            # Normalize models between clients...
            if self.normalize_dec:
                normalization_term = np.amax(my_client.w)
            else:
                normalization_term = 1
            aggr_w += (my_client.learning_batch/summed_num_datapoints) * my_client.w / normalization_term
        # Normalize the resulting global model
        if self.normalize_dec:
            aggr_w /= np.amax(aggr_w)
        ###########################################################################################################################
        
        # This would be the place to do smoothbatch if we wanted to do it on a global level
        # Right now the global decoders are essentially independent
        
        #print(np.sum(self.w-aggr_w))
        #assert(np.sum(self.w-aggr_w) != 0)
        self.w = aggr_w
        
    # Function for getting the finalized personalized and global model 
    def set_finalized_APFL_models(self):
        # Defining params used in the next section
        for t in range(T):
            summed_chosen_globals = np.sum([chosen_client.global_w for chosen_client in self.chosen_clients_lst])
            for my_client in self.available_clients_lst:
                my_client.running_pers_term += my_client.p[t]*(my_client.adap_alpha*my_client.local_w + (1-my_client.adap_alpha)*(1/K)*summed_chosen_globals) 
                my_client.running_global_term += my_client.p[t]*summed_chosen_globals
        # The actual models                                                    
        # "tin for i=1,...,n"
        for my_client in self.available_clients_lst:
            S_T = np.sum(my_client.p)
            # All these random params are defined in theorem 2 on Page 10
            # "Output" --> lol and do what with
            # Personalized model: v^hat AKA personalized_w = (1/S_T)*\sum_1^T(p_t(alpha_i*v_i^t + (1-alpha_i)*(1/K)*(\sum_{j in chosen clients} w_j^t)))
            my_client.final_personalized_w = (1/S_T)*my_client.running_pers_term
            # Global model: w^hat = 1/(K*S_T)*(\sum_1^T p_t*(\sum_j w_j^t))
            my_client.final_global_w = (1/K*S_T)*my_client.running_global_term
