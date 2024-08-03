import numpy as np
import random
import copy
#from matplotlib import pyplot as plt
#import time
#import pickle

from experiment_params import *
from cost_funcs import *
from fl_sim_base import *
        
        
class Server(ModelBase):
    def __init__(self, ID, D0, opt_method, global_method, all_clients, smoothbatch=0.75, C=0.35, normalize_dec=False, test_split_type='end', 
                 test_split_frac=0.3, current_round=0, PCA_comps=64, verbose=False, validate_memory_IDs=True):
        super().__init__(ID, D0, opt_method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, 
                         verbose=verbose, num_participants=14, log_init=0)
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
        self.validate_memory_IDs = validate_memory_IDs
        self.test_split_type = test_split_type
        self.test_split_frac = test_split_frac
        self.global_method = global_method.upper()
        print(f"Running the {self.global_method} algorithm as the global method!")
        self.set_available_clients_list()

                
    # 0: Main Loop
    def execute_FL_loop(self):
        # Update global round number
        self.current_round += 1
        
        # TODO: This is supposed to include PerFedAvg I'm assuming...
        if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
            # Choose fraction C of available clients
            self.set_available_clients_list()
            self.choose_clients()
            # Let those clients train (this autoselects the chosen_client_lst to use)
            self.train_client_and_log(training_client_set=self.chosen_clients_lst)
            # AGGREGATION
            self.w_prev = copy.deepcopy(self.w)
            self.agg_local_weights()  # This func sets self.w, eg the new decoder
            # GLOBAL SmoothBatch
            #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
            #^ Note when self.smoothbatch=1 (default), we just keep the new self.w (no mixing)
            if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
                self.w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.w_prev)
                # Eg don't do a global-global smoothbatch for the other cases
        elif self.global_method=='NOFL':
            # TODO: Is NoFL just supposed to be the Local CPHS sims... if so this is fine I think 
            self.train_client_and_log(client_set=self.all_clients)
        else:
            raise('Method not currently supported, please reset method to FedAvg')
        
        # Save the new decoder to the log
        self.dec_log.append(copy.deepcopy(self.w))
        # Run test_metrics to generate performance on testing data
        for client_idx, my_client in enumerate(self.available_clients_lst):
            # Reset all clients so no one is chosen for the next round
            if type(my_client)==int:
                raise TypeError("All my clients are all integers...")
            my_client.chosen_status = 0
            # test_metrics for all clients
            if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
                global_test_loss, _ = my_client.test_metrics(self.w, 'global')
                local_test_loss, _ = my_client.test_metrics(my_client.w, 'local')
            elif self.global_method=='NOFL':
                global_test_loss = 0
                local_test_loss, _ = my_client.test_metrics(my_client.w, 'local') 
            
            # TODO: Why are these arrays and not just adding the sums? ...
            if client_idx!=0:
                running_global_test_loss += np.array(global_test_loss)
                running_local_test_loss += np.array(local_test_loss)
            else:
                running_global_test_loss = np.array(global_test_loss)
                running_local_test_loss = np.array(local_test_loss)

        if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
            # TODO: I think this is averaged incorrectly... loss should be averaged over num_samples not num_clients I think?
            # TODO: THIS IS TEST METRICS which i called on available_clients_lst so I think self.all_cavailable_clients_lstlients is fine?
            self.global_test_error_log = running_global_test_loss / len(self.available_clients_lst)
            self.local_test_error_log = running_local_test_loss / len(self.available_clients_lst)
        elif self.global_method=='NOFL':
            self.local_test_error_log = running_local_test_loss / len(self.available_clients_lst)
            
    # 1.1
    def set_available_clients_list(self):
        self.num_avail_clients = 0
        self.available_clients_full_idx_lst = [0]*len(self.all_clients)
        for idx, my_client in enumerate(self.all_clients):
            if my_client.availability:
                self.available_clients_full_idx_lst[idx] = my_client
                self.num_avail_clients += 1
        self.available_clients_lst = [val for val in self.available_clients_full_idx_lst if val != 0]
    
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
    def train_client_and_log(self, training_client_set):
        current_local_lst = []
        current_global_lst = []
        current_pers_lst = []
        for my_client in self.all_clients:  
            if my_client.availability==False:
                # If this doesnt append here then current_local_lst size doesnt change (from train_clients)
                ## Which is why local_error_log[-1] size doesn't change (eg is still train_clients)
                # TODO: Make sure that any averaging does not take these into account...
                ## Make sure that only available clients are plotted or whatever...
                ## Sounds like a pain for sequential.......
                current_local_lst.append((my_client.ID, self.current_round, 0))
                if self.global_method != 'NOFL':
                    current_global_lst.append((my_client.ID, self.current_round, global_init_carry_val))
                if self.global_method in self.pers_methods:
                    current_pers_lst.append((my_client.ID, self.current_round, pers_init_carry_val))
                continue

            # ^ Implications of using available_clients_lst instead of all_clients?
            # ... Or should it be training_clients for cross_val...
            my_client.latest_global_round = self.current_round
            #^ Need to overwrite client with the current global round, for t later
            # ^ Idk what this comment is about...
            
            # This isn't great code because it checks the init every single time it runs
            #  Maybe move this to be before this loop?
            if len(self.local_error_log)==0:  #self.local_error_log[-1]==self.log_init:
                local_init_carry_val = 0
                global_init_carry_val = 0
                pers_init_carry_val = 0
            else:
                local_init_carry_val = self.local_error_log[-1][my_client.ID][2]#[0]
                if self.global_method != 'NOFL':
                    global_init_carry_val = self.global_error_log[-1][my_client.ID][2]#[0]
                if self.global_method in self.pers_methods:
                    pers_init_carry_val = self.pers_error_log[-1][my_client.ID][2]#[0]
                    
            if my_client in training_client_set:
                # Send those clients the current global model
                my_client.global_w = copy.deepcopy(self.w)

                my_client.execute_training_loop()
                
                if self.validate_memory_IDs:
                    assert(id(my_client.w)!=id(self.w))
                    assert(id(my_client.w)!=id(my_client.global_w))
                
                current_local_lst.append((my_client.ID, self.current_round, 
                                          my_client.eval_model(which='local')))
                if self.global_method != 'NOFL':
                    current_global_lst.append((my_client.ID, self.current_round, 
                                               my_client.eval_model(which='global')))
                # TODO: Does this do anything... how is this different from the local model...
                if self.global_method in self.pers_methods:
                    current_pers_lst.append((my_client.ID, self.current_round, 
                                               my_client.eval_model(which='pers')))
            else:
                current_local_lst.append((my_client.ID, self.current_round, 
                                          local_init_carry_val))
                if self.global_method != 'NOFL':
                    current_global_lst.append((my_client.ID, self.current_round,
                                               global_init_carry_val))
                if self.global_method in self.pers_methods:
                    current_pers_lst.append((my_client.ID, self.current_round,
                                               pers_init_carry_val))
        # Append (ID, COST) to SERVER'S error log.  
        #  Note that round is implicit, it is just the index of the error log
        if self.local_error_log==self.init_lst:
            # TODO: Remove this code? ...
            # Overwrite the [(0,0)] hold... why not just init to this then...
            #self.local_error_log = []
            self.local_error_log.append(current_local_lst)
        else:
            self.local_error_log.append(current_local_lst)
        if self.global_method != 'NOFL':
            # NoFL case has no global model since there's... no FL
            if self.global_error_log==self.init_lst:
                self.global_error_log = []
                self.global_error_log.append(current_global_lst)
            else:
                self.global_error_log.append(current_global_lst)
        if self.global_method in self.pers_methods:
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
        
