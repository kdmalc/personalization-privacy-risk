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


class ModelBase:
    # Hard coded attributes --> SHARED FOR THE ENTIRE CLASS
    # ^Are they? I'm not actually sure.  You can't access them obviously
    num_updates = 19
    cphs_starting_update = 10
    update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    
    def __init__(self, ID, w, method, smoothbatch=1, verbose=False, PCA_comps=7, current_round=0, num_participants=14, log_init=0):
        self.type = 'BaseClass'
        self.ID = ID
        self.PCA_comps = PCA_comps
        self.pca_channel_default = 64  # When PCA_comps equals this, DONT DO PCA
        if w.shape!=(2, self.PCA_comps):
            print(f"Class BaseModel: Overwrote the provided init decoder: {w.shape} --> {(2, self.PCA_comps)}")
            self.w = np.random.rand(2, self.PCA_comps)
        else:
            self.w = w
        self.w_prev = copy.deepcopy(self.w)
        self.dec_log = [copy.deepcopy(self.w)]
        self.w_prev = copy.deepcopy(self.w)
        self.num_participants = num_participants
        self.log_init = log_init
        self.local_error_log = [log_init]*num_participants
        self.global_error_log = [log_init]*num_participants
        self.pers_error_log = [log_init]*num_participants
        self.method = method
        self.current_round = current_round
        self.verbose = verbose
        self.smoothbatch = smoothbatch
        self.pers_methods = ['FedAvgSB', 'APFL', 'Per-FedAvg FO', 'Per-FedAvg HF']

        
    def __repr__(self): 
        return f"{self.type}{self.ID}"
    
    def display_info(self): 
        return f"{self.type} model: {self.ID}\nCurrent Round: {self.current_round}\nTraining Method: {self.method}"


class TrainingMethods:
    # Different training approaches
    
    # This one blows up to NAN/overflow... not sure why
    def train_eta_gradstep(self, w, eta, F, D, H, V, learning_batch, alphaF, alphaD, PCA_comps):
        grad_cost = np.reshape(gradient_cost_l2(F, D, H, V, learning_batch, alphaF, alphaD, Ne=PCA_comps),(2, PCA_comps))
        w_new = w - eta*grad_cost
        return w_new

    def train_eta_scipyminstep(self, w, eta, F, D, H, V, learning_batch, alphaF, alphaD, D0, display_info, PCA_comps, full=False):
        # I turned off display_info because it's kind of annoying
        if full:
            out = minimize(lambda D: cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps), D0, method='BFGS', jac=lambda D: gradient_cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps))#, options={'disp': display_info})
        else:
            out = minimize(lambda D: cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps), D0, method='BFGS', jac=lambda D: gradient_cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps), options={'maxiter':eta}) #'disp': display_info, 
        w_new = np.reshape(out.x,(2, PCA_comps))
        return w_new
        
        
class Server(ModelBase):
    def __init__(self, ID, D0, method, all_clients, smoothbatch=1, C=0.1, normalize_dec=True, current_round=0, PCA_comps=7, verbose=False, APFL_Tau=10):
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
        if self.method=='APFL':
            self.set_available_clients_list()
            self.choose_clients()
            self.K = len(self.chosen_clients_lst)
            self.tau = APFL_Tau

                
    # 0: Main Loop
    def execute_FL_loop(self):
        # Update global round number
        self.current_round += 1
        
        #if self.method in ['FedAvg', 'FedAvgSB']:
        if 'FedAvg' in self.method:
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
                self.w = (1/self.K)*copy.deepcopy(running_global_dec)
                self.set_available_clients_list()
                self.choose_clients()
                self.K = len(self.chosen_clients_lst)
                # Presumably len(self.chosen_clients_lst) will always be the same (same C)... 
                # otherwise would need to re-set K as so:
                #self.K = len(self.chosen_clients_lst)

                for my_client in self.chosen_clients_lst:
                    my_client.global_w = copy.deepcopy(self.w)  
        else:
            raise('Method not currently supported, please reset method to FedAvg')
        # Save the new decoder to the log
        self.dec_log.append(copy.deepcopy(self.w))
        # Reset all clients so no one is chosen for the next round
        for my_client in self.all_clients:  # Would it be better to use just the chosen_clients or do all_clients?
            if type(my_client)==int:
                raise TypeError("All my clients are all integers...")
            my_client.chosen_status = 0
        
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
            my_client.current_global_round = self.current_round
            #^ Need to overwrite client with the curernt global round, for t later
            
            # This isn't great code because it checks the init every single time it runs
            #  Maybe move this to be before this loop?
            if self.local_error_log[-1]==self.log_init:
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
                my_client.global_w = self.w
                
                my_client.execute_training_loop()
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
        aggr_w = 0
        for my_client in self.chosen_clients_lst:
            # Normalize models between clients...
            if self.normalize_dec:
                normalization_term = np.amax(my_client.w)
            else:
                normalization_term = 1
            aggr_w += (my_client.learning_batch/summed_num_datapoints) * my_client.w / normalization_term
        # Normalize the resulting global model
        if self.normalize_dec:
            aggr_w /= np.amax(aggr_w)
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
                                                               

class Client(ModelBase, TrainingMethods):
    def __init__(self, ID, w, method, local_data, data_stream, smoothbatch=1, current_round=0, PCA_comps=7, availability=1, global_method='FedAvg', normalize_dec=False, normalize_EMG=True, track_cost_components=True, track_lr_comps=True, use_real_hess=True, gradient_clipping=False, log_decs=True, clipping_threshold=100, tol=1e-10, adaptive=True, eta=1, track_gradient=True, wprev_global=False, num_steps=1, use_zvel=False, APFL_input_eta=False, safe_lr_factor=False, mix_in_each_steps=False, mix_mixed_SB=False, delay_scaling=5, random_delays=False, download_delay=1, upload_delay=1, local_round_threshold=25, condition_number=1, verbose=False):
        super().__init__(ID, w, method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, verbose=verbose, num_participants=14, log_init=0)
        '''
        Note self.smoothbatch gets overwritten according to the condition number!  
        If you want NO smoothbatch then set it to 'off'
        '''
        # NOT INPUT
        self.type = 'Client'
        self.chosen_status = 0
        self.latest_global_round = 0
        self.update_transition_log = []
        self.normalize_EMG = normalize_EMG
        # Sentinel Values
        self.F = None
        self.V = None
        self.H = np.zeros((2,2))
        self.learning_batch = None
        self.dt = 1.0/60.0
        self.eta = eta  # Learning rate
        self.training_data = local_data['training']
        self.labels = local_data['labels']
        # Round minimization output to the nearest int or keep as a float?  Don't need arbitrary precision
        self.round2int = False
        self.normalize_dec = normalize_dec
        ####################################################################
        # Maneeshika Code:
        self.use_zvel = use_zvel
        self.hit_bound = 0
        ####################################################################
        # FL CLASS STUFF
        # Availability for training
        self.availability = availability
        # Toggle streaming aspect of data collection: {Ignore updates and use all the data; 
        #  Stream each update, moving to the next update after local_round_threshold iters have been run; 
        #  After 1 iteration, move to the next update}
        self.data_stream = data_stream  # {'full_data', 'streaming', 'advance_each_iter'} 
        # Number of gradient steps to take when training (eg amount of local computation)
        self.num_steps = num_steps
        self.wprev_global = wprev_global
        # GLOBAL STUFF
        self.global_method = global_method
        # UPDATE STUFF
        if self.global_method=='NoFL':
            starting_update = 0
        else:
            starting_update = self.cphs_starting_update
        self.current_update = starting_update
        self.local_round_threshold = local_round_threshold
        #
        # Not even using the delay stuff right now
        # Boolean setting whether or not up/download delays should be random or predefined
        self.random_delays = random_delays
        # Scaling from random [0,1] to number of seconds
        self.delay_scaling = delay_scaling
        # Set the delay times
        if self.random_delays: 
            self.download_delay = random.random()*self.delay_scaling
            self.upload_delay = random.random()*self.delay_scaling
        else:
            self.download_delay = download_delay
            self.upload_delay = upload_delay
        #
        # ML Parameters / Conditions        
        cond_dict = {1:(0.25, 1e-3, 1), 2:(0.25, 1e-4, 1), 3:(0.75, 1e-3, 1), 4:(0.75, 1e-4, 1), 5:(0.25, 1e-4, -1), 6:(0.25, 1e-4, -1), 7:(0.75, 1e-3, -1), 8:(0.75, 1e-4, -1)}
        cond_smoothbatch, self.alphaD, self.init_dec_sign = cond_dict[condition_number]
        if type(smoothbatch)==str and smoothbatch.upper()=='OFF':
            self.smoothbatch = 1  # AKA Use only the new dec, no mixing
        elif smoothbatch==1:  # This is the default
            # If it is default, then let the condition number set smoothbatch
            self.smoothbatch = cond_smoothbatch
        else:
            # Set smoothbatch to whatever you manually entered
            self.smoothbatch=smoothbatch
            print()
        self.alphaE = 1e-6
        self.alphaF = 1e-7
        #
        self.gradient_clipping = gradient_clipping
        self.clipping_threshold = clipping_threshold
        # PLOTTING
        self.log_decs = log_decs
        self.pers_dec_log = [np.zeros((2,self.PCA_comps))]
        self.global_dec_log = [np.zeros((2,self.PCA_comps))]
        # Overwrite the logs since global and local track in slightly different ways
        self.local_error_log = []
        self.global_error_log = []
        self.pers_error_log = []
        self.track_cost_components = track_cost_components
        self.performance_log = []
        self.Dnorm_log = []
        self.Fnorm_log = []
        self.track_gradient = track_gradient
        self.gradient_log = []
        self.pers_gradient_log = []
        self.global_gradient_log = []
        # FedAvgSB Stuff
        self.mix_in_each_steps = mix_in_each_steps
        self.mix_mixed_SB = mix_mixed_SB
        self.APFL_input_eta = APFL_input_eta  # Is this really an APFL thing only?
        # These are general personalization things
        self.running_pers_term = 0
        self.running_global_term = 0
        self.global_w = copy.deepcopy(self.w)
        self.mixed_w = copy.deepcopy(self.w)
        # APFL Stuff
        self.tol = tol 
        self.track_lr_comps = track_lr_comps
        self.L_log = []
        self.mu_log = []
        self.eta_t_log = []
        self.adaptive = adaptive
        #They observed best results with 0.25, but initialized adap_alpha to 0.01 for the adaptive case
        if self.adaptive:
            self.adap_alpha = [0.01]  
        else:
            self.adap_alpha = [0.25] 
        self.tau = self.num_steps # This is just an init... it really ought to pull it from global...
        self.p = [0]
        self.safe_lr_factor = safe_lr_factor
        self.Vmixed = None
        self.Vglobal = None
        self.use_real_hess = use_real_hess
        self.prev_eigvals = None
            
            
    # 0: Main Loop
    def execute_training_loop(self):
        self.simulate_data_stream()
        self.train_model()
        
        # LOG EVERYTHING
        # Log decs
        if self.log_decs:
            self.dec_log.append(self.w)
            if self.global_method=="FedAvg":
                self.global_dec_log.append(self.global_w)
            elif self.global_method in self.pers_methods:
                self.global_dec_log.append(self.global_w)
                self.pers_dec_log.append(self.mixed_w)
        # Log Error
        self.local_error_log.append(self.eval_model(which='local'))
        # Yes these should both be ifs not elif, they may both need to run
        if self.global_method!="NoFL":
            self.global_error_log.append(self.eval_model(which='global'))
        if self.global_method in self.pers_methods:
            self.pers_error_log.append(self.eval_model(which='pers'))
        # Log Cost Comp
        if self.track_cost_components:
            if self.global_method=='APFL':
                # It is using self.V here for vplus, Vminus... not sure if that is correct
                self.performance_log.append(self.alphaE*(np.linalg.norm((self.mixed_w@self.F + self.H@self.V[:,:-1] - self.V[:,1:]))**2))
                self.Dnorm_log.append(self.alphaD*(np.linalg.norm(self.mixed_w)**2))
                self.Fnorm_log.append(self.alphaF*(np.linalg.norm(self.F)**2))
            else:
                self.performance_log.append(self.alphaE*(np.linalg.norm((self.w@self.F + self.H@self.V[:,:-1] - self.V[:,1:]))**2))
                self.Dnorm_log.append(self.alphaD*(np.linalg.norm(self.w)**2))
                self.Fnorm_log.append(self.alphaF*(np.linalg.norm(self.F)**2))
        # Log For APFL Gradient... why have this be separate from the rest...
        # Wouldn't I also want to try and log the global and pers gradients?
        if self.track_gradient==True and self.global_method!="APFL":
            # The gradient is a vector... So let's just save the L2 norm?
            self.gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, self.w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)))

        
    def simulate_delay(self, incoming):
        if incoming:
            time.sleep(self.download_delay+random.random())
        else:
            time.sleep(self.upload_delay+random.random())
            
            
    def simulate_data_stream(self, streaming_method=True):
        if streaming_method:
            streaming_method = self.data_stream
        need_to_advance=True
        self.current_round += 1
        if self.current_update==16:  #17: previously 17 but the last update is super short so I cut it out
            #print("Maxxed out your update (you are on update 18), continuing training on last update only")
            # Probably ought to track that we maxed out --> LOG SYSTEM
            # We are stopping an update early, so use -3/-2 and not -2/-1 (the last update)
            lower_bound = (update_ix[-3] + update_ix[-2])//2  #Use only the second half of each update
            upper_bound = update_ix[-2]
            self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='full_data':
            lower_bound = update_ix[0]  # Starts at 0 and not update 10, for now
            upper_bound = update_ix[-1]
            self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='streaming':
            # If we pass threshold, move on to the next update
            if self.current_round%self.local_round_threshold==0:
                self.current_update += 1
                
                self.update_transition_log.append(self.latest_global_round)
                if self.verbose==True and self.ID==1:
                    print(f"Client {self.ID}: New update after lrt passed: (new update, current global round, current local round): {self.current_update, self.latest_global_round, self.current_round}")
                    print()
                    
                # Using only the second half of each update for co-adaptivity reasons
                lower_bound = (update_ix[self.current_update] + update_ix[self.current_update+1])//2  
                upper_bound = update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
            elif self.current_round>2:
                # This is the base case
                # The update number didn't change so we don't need to overwrite everything with the same data
                need_to_advance = False
            else:
                # This is for the init case (current round is 0 or 1)
                # need_to_advance is true, so we overwrite s and such... this is fine 
                lower_bound = (update_ix[self.current_update] + update_ix[self.current_update+1])//2  
                upper_bound = update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='advance_each_iter':
            lower_bound = (update_ix[self.current_update] + update_ix[self.current_update+1])//2  
            upper_bound = update_ix[self.current_update+1]
            self.learning_batch = upper_bound - lower_bound
            
            self.current_update += 1
        else:
            raise ValueError(f'streaming_method ("{streaming_method}") not recognized: this data streaming functionality is not supported')
            
        if need_to_advance:
            s_temp = self.training_data[lower_bound:upper_bound,:]
            # First, normalize the entire s matrix
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            # Now do PCA unless it is set to 64 (AKA the default num channels i.e. no reduction)
            # Also probably ought to find a global transform if possible so I don't recompute it every time...
            if self.PCA_comps!=self.pca_channel_default:  
                pca = PCA(n_components=self.PCA_comps)
                s_normed = pca.fit_transform(s_normed)
            s = np.transpose(s_normed)
            self.F = s[:,:-1] # note: truncate F for estimate_decoder
            v_actual = self.w@s
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            p_reference = np.transpose(self.labels[lower_bound:upper_bound,:])
            
            #####################################################################
            # Add the boundary conditions code here
            if self.use_zvel:
                # Maneeshika code
                p_ref_lim = self.labels[lower_bound:upper_bound,:]
                if self.current_round<2:
                    self.vel_est = np.zeros_like((p_ref_lim))
                    self.pos_est = np.zeros_like((p_ref_lim))
                    self.int_vel_est = np.zeros_like((p_ref_lim))
                    self.vel_est[0] = self.w@s[:,0]  # Translated from: Ds_fixed@emg_tr[0]
                    self.pos_est[0] = [0, 0]
                else:
                    prev_vel_est = self.vel_est[-1]
                    prev_pos_est = self.pos_est[-1]
                    
                    self.vel_est = np.zeros_like((p_ref_lim))
                    self.pos_est = np.zeros_like((p_ref_lim))
                    self.int_vel_est = np.zeros_like((p_ref_lim))
                    
                    self.vel_est[0] = prev_vel_est
                    self.pos_est[0] = prev_pos_est
                for tt in range(1, s.shape[1]):
                    # Note this does not keep track of actual updates, only the range of 1 to s.shape[1] (1202ish)
                    vel_plus = self.w@s[:,tt]  # Translated from: Ds_fixed@emg_tr[tt]
                    p_plus = self.pos_est[tt-1, :] + (self.vel_est[tt-1, :]*self.dt)
                    # These are just correctives, such that vel_plus can get bounded
                    # x-coordinate
                    if abs(p_plus[0]) > 36:  # 36 hardcoded from earlier works
                        p_plus[0] = self.pos_est[tt-1, 0]
                        vel_plus[0] = 0
                        self.hit_bound += 1 # update hit_bound counter
                    if abs(p_plus[1]) > 24:  # 24 hardcoded from earlier works
                        p_plus[1] = self.pos_est[tt-1, 1]
                        vel_plus[1] = 0
                        self.hit_bound += 1 # update hit_bound counter
                    if self.hit_bound > 200:  # 200 hardcoded from earlier works
                        p_plus[0] = 0
                        vel_plus[0] = 0
                        p_plus[1] = 0
                        vel_plus[1] = 0
                        self.hit_bound = 0
                    # now update velocity and position
                    self.vel_est[tt] = vel_plus
                    self.pos_est[tt] = p_plus
                    # calculate intended velocity
                    self.int_vel_est[tt] = calculate_intended_vels(p_ref_lim[tt], p_plus, 1/self.dt)

                self.V = np.transpose(self.int_vel_est[:tt+1])
                #print(f"V.shape: {self.V.shape}")
            else:
                # Original code
                self.V = (p_reference - p_actual)*self.dt
            
            if self.global_method=='APFL':
                self.Vglobal = (p_reference - np.cumsum(self.global_w@s, axis=1)*self.dt)*self.dt
                #self.Vlocal = (p_reference - np.cumsum(self.w@s, axis=1)*self.dt)*self.dt 
                # ^Here, Vlocal is just self.V! Same eqn
                # ^Should this be local or mixed? I think local... 
                # ^Even though it is evaluated at the mixed dec... not sure
                # For V that is used with mixed I think it should actually be mixed.  Makes more sense
                self.Vmixed = (p_reference - np.cumsum(self.mixed_w@s, axis=1)*self.dt)*self.dt
    
    
    def train_model(self):
        D_0 = copy.copy(self.w_prev)
        # Set the w_prev equal to the current w:
        self.w_prev = copy.copy(self.w)
        if self.global_method in ["FedAvg", "NoFL", "FedAvgSB", "Per-FedAvg", "Per-FedAvg FO", "Per-FedAvg HF"]:
            if self.global_method!="NoFL":
                # Overwrite local model with the new global model
                self.w = copy.deepcopy(self.global_w)
            
            for i in range(self.num_steps):
                # I think this ought to be on but it makes the global model and gradient diverge...
                # Why can't i do this outside the loop
                if self.wprev_global==True and i==0 and ('Per-FedAvg' in self.method):
                    self.w_prev = copy.deepcopy(self.global_w)
                
                ########################################
                # Should I normalize the dec here?  
                # I think this will prevent it from blowing up if I norm it every time
                if self.normalize_dec:
                    self.w /= np.amax(self.w)
                ########################################
                if self.method=='EtaGradStep':
                    self.w = self.train_eta_gradstep(self.w, self.eta, self.F, self.w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, PCA_comps=self.PCA_comps)
                elif self.method=='EtaScipyMinStep':
                    self.w = self.train_eta_scipyminstep(self.w, self.eta, self.F, self.w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, D_0, self.verbose, PCA_comps=self.PCA_comps)
                elif self.method=='FullScipyMinStep':
                    self.w = self.train_eta_scipyminstep(self.w, self.eta, self.F, self.w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, D_0, self.verbose, PCA_comps=self.PCA_comps, full=True)
                elif self.method=='Per-FedAvg':
                    raise("Per-FedAvg uses the Hessian which is sus, choose the <Per-FedAvg FO> or <Per-FedAvg HF> approximations")
                    self.w_tilde = self.w_prev - self.eta * gradient_cost_l2(self.F, self.w_prev, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                    self.w = self.w_prev - self.beta*(np.identity(1) - self.alpha*hessian_cost_l2(self.F, self.alphaD)) * gradient_cost_l2(self.F, self.w_prev, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                elif self.method=='Per-FedAvg FO':
                    self.w_tilde = self.w_prev - self.eta * gradient_cost_l2(self.F, self.w_prev, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                    self.w = gradient_cost_l2(self.F, self.w_tilde, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                elif self.method=='Per-FedAvg HF':
                    # Difference of gradients method
                    alpha = self.eta  # Yes
                    delta = self.eta  # Not sure... not listed in paper
                    # w inside original gradient for MAML
                    w_tilde = self.w_prev - alpha * gradient_cost_l2(self.F, self.w_prev, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                    # First gradient term (+)
                    grad1 = gradient_cost_l2(self.F, (self.w_prev + delta*gradient_cost_l2(self.F, (self.w_prev - alpha*gradient_cost_l2(self.F, self.w_prev, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)), self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)), self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                    # Second gradient term (flipped to -)
                    grad2 = gradient_cost_l2(self.F, (self.w_prev - delta*gradient_cost_l2(self.F, (self.w_prev - alpha*gradient_cost_l2(self.F, self.w_prev, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)), self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)), self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False)
                    # Computing d_i(w)
                    d = (grad1 - grad2)/(2*delta)
                    # Set current weight based on the above
                    self.w = gradient_cost_l2(self.F, w_tilde, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps, flatten=False) - alpha*d
                else:
                    raise ValueError("Unrecognized method")
                if self.mix_in_each_steps:
                    self.mixed_w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.mixed_w)
            ########################################
            # Or should I normalize the dec here?  I'll also turn this on since idc about computational speed rn
            if self.normalize_dec:
                self.w /= np.amax(self.w)
            ########################################
            # Do SmoothBatch
            # Maybe move this to only happen after each update? Does it really need to happen every iter?
            # I'd have to add weird flags just for this in various places... put on hold for now
            #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
            if self.global_method in ["FedAvg", "NoFL"]:  # Maybe should add Per-FedAvg here...
                self.w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.w_prev)
            elif self.global_method=="FedAvgSB":
                global_local_SB = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.global_w)
                if self.mix_mixed_SB:
                    self.mixed_w = self.smoothbatch*self.mixed_w + ((1 - self.smoothbatch)*global_local_SB)
                else:
                    self.mixed_w = global_local_SB
        elif self.global_method=='APFL': 
            t = self.latest_global_round  # Should this be global or local? Global based on how they wrote it...
            # eig is for unsymmetric matrices, and returns (UNORDERED) eigvals, eigvecs
            if self.use_real_hess:
                if t < 2:
                    eigvals, _ = np.linalg.eig(hessian_cost_l2(self.F, self.alphaD))
                    self.prev_eigvals = eigvals
                elif self.latest_global_round not in self.update_transition_log:
                    # Note that this should not be run if you want to do SGD instead of GD
                    # Eg probably need to change the logic structure
                    eigvals = self.prev_eigvals
                else:
                    print(f"Client{self.ID}: Recalculating the Hessian for new update {self.current_update}!")
                    eigvals, _ = np.linalg.eig(hessian_cost_l2(self.F, self.alphaD))
                    self.prev_eigvals = eigvals
            else:
                # Can try and add faster versions in the future
                raise ValueError("Currently, the only option is to use the Real Hessian")
            
            mu = np.amin(eigvals)  # Mu is the minimum eigvalue
            if mu==None:
                print(f"mu: {mu}")
                print(f"eigvals: {eigvals}")
                raise ValueError("mu is None for some reason")
            if mu.imag < self.tol and mu.real < self.tol:
                raise ValueError("mu is ~0, thus implying func is not mu-SC")
            elif mu.imag < self.tol:
                mu = mu.real
            elif mu.real < self.tol:
                print("Setting to imaginary only")  # This is an issue if this runs
                mu = mu.imag
                
            L = np.amax(eigvals)  # L is the maximum eigvalue
            if L==None:
                print(f"L: {L}")
                print(f"eigvals: {eigvals}")
                raise ValueError("L is None for some reason")
            if L.imag < self.tol and L.real < self.tol:
                raise ValueError("L is 0, thus implying func is not L-smooth")
            elif mu.imag < self.tol:
                L = L.real
            elif L.real < self.tol:
                print("Setting to imaginary only")  # This is an issue if this runs
                L = L.imag
            if self.verbose: 
                # Find a better way to print this out without spamming the console... eg log file...
                print(f"Client{self.ID}: L: {L}, mu: {mu}")
            kappa = L/mu
            a = np.max([128*kappa, self.tau])
            eta_t = 16 / (mu*(t+a))
            if self.APFL_input_eta:
                if self.safe_lr_factor!=False:
                    raise ValueError("Cannot use APFL_input_eta AND safe_lr_factor (they overwrite each other)")
                eta_t = self.eta
            elif self.safe_lr_factor!=False:
                print("Forcing eta_t to be based on the input safe lr factor")
                # This is only subtly different from just inputting eta... a little more dynamic ig
                eta_t = 1/(self.safe_lr_factor*L)
            elif eta_t >= 1/(2*L):
                # Note that we only check when automatically setting
                # ie if you manually input it will do whatever you tell it to do
                raise ValueError("Learning rate is too large according to constaints on GD")
            if self.verbose:
                print(f"Client{self.ID}: eta_t: {eta_t}")
            self.p.append((t+a)**2)
            if self.track_lr_comps:
                self.L_log.append(L)
                self.mu_log.append(mu)
                self.eta_t_log.append(eta_t)
            
            if self.adaptive:
                self.adap_alpha.append(self.adap_alpha[-1] - eta_t*np.inner(np.reshape((self.w-self.global_w), (self.PCA_comps*2)), np.reshape(gradient_cost_l2(self.F, self.mixed_w, self.H, self.Vmixed, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps), (2*self.PCA_comps))))
                # This is theoretically the same but I'm not sure what grad_alpha means
                #self.sus_adap_alpha.append() ... didn't write yet

            # GRADIENT DESCENT BASED MODEL UPDATE
            # NOTE: eta_t IS DIFFERENT FROM CLIENT'S ETA (WHICH IS NOT USED)            
            global_gradient = np.reshape(gradient_cost_l2(self.F, self.global_w, self.H, self.Vglobal, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps), (2, self.PCA_comps))
            local_gradient = np.reshape(gradient_cost_l2(self.F, self.mixed_w, self.H, self.Vmixed, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps), (2, self.PCA_comps))
            # Gradient clipping
            if self.gradient_clipping:
                if np.linalg.norm(global_gradient) > self.clipping_threshold:
                    global_gradient = self.clipping_threshold*global_gradient/np.linalg.norm(global_gradient)
                if np.linalg.norm(local_gradient) > self.clipping_threshold:
                    local_gradient = self.clipping_threshold*local_gradient/np.linalg.norm(local_gradient)
                
            ########################################
            # Or should I normalize the dec here?  I'll also turn this on since idc about computational speed rn
            if self.normalize_dec:
                self.global_w /= np.amax(self.global_w)
                self.w /= np.amax(self.w)
                self.mixed_w /= np.amax(self.mixed_w)
            ########################################
            
            # PSEUDOCODE: my_client.global_w -= my_client.eta * grad(f_i(my_client.global_w; my_client.smallChi))
            self.global_w -= eta_t * global_gradient
            # PSEUDOCODE: my_client.local_w -= my_client.eta * grad_v(f_i(my_client.v_bar; my_client.smallChi))
            self.w -= eta_t * local_gradient
            self.mixed_w = self.adap_alpha[-1]*self.w - (1 - self.adap_alpha[-1])*self.global_w
            ########################################
            # Or should I normalize the dec here?  I'll also turn this on since idc about computational speed rn
            if self.normalize_dec:
                self.global_w /= np.amax(self.global_w)
                self.w /= np.amax(self.w)
                self.mixed_w /= np.amax(self.mixed_w)
            ########################################
            
        # Save the new decoder to the log
        #self.dec_log.append(self.w)
        #if self.global_method in self.pers_methods:
        #    self.pers_dec_log.append(self.mixed_w)
        #self.global_dec_log.append(self.global_w)
        
        # Logging the grad here and in exec was causing the muted gradient bumps
        if self.global_method=="APFL" and self.track_gradient==True:
            # FOR APFL ONLY
            self.gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, self.w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)))
            self.pers_gradient_log.append(np.linalg.norm(local_gradient))
            # ^Local_gradient is evaluated wrt mixed inputs (eg w and V) so it's the pers gradient here
            self.global_gradient_log.append(np.linalg.norm(global_gradient))
            # Sceptical about the validity of the global and pers gradients in these cases
            #elif "FedAvg" in self.global_method:
            #    # Not sure if V is correct here... need to use a global V?
            #    self.global_gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, self.global_w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)))
            #    if "SB" in self.global_method:
            #        # Also not sure about V here...
            #        self.pers_gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, self.mixed_w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)))
        
        
    def eval_model(self, which):
        if which=='local':
            my_dec = self.w
            my_V = self.V
        elif which=='global':
            my_dec = self.global_w
            # self.V for non APFL case, only APFL defines Vglobal, as of 3/21
            my_V = self.Vglobal if self.global_method=='APFL' else self.V
        elif which=='pers' and self.global_method in self.pers_methods:
            my_dec = self.w if 'Per-FedAvg' in self.global_method else self.mixed_w
            # self.V for non APFL case, only APFL defines Vmixed, as of 3/24
            my_V = self.Vmixed if self.global_method=='APFL' else self.V
        else:
            raise ValueError("Please set <which> to either local or global")
        # Just did this so we wouldn't have the 14 decimals points it always tries to give
        if self.round2int:
            temp = np.ceil(cost_l2(self.F, my_dec, self.H, my_V, self.learning_batch, self.alphaF, self.alphaD))
            # Setting to int is just to catch overflow errors
            # For RT considerations, ints are also generally ints cheaper than floats...
            out = int(temp)
        else:
            temp = cost_l2(self.F, my_dec, self.H, my_V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)
            out = round(temp, 3)
        return out
        
    def test_inference(self, test_current_dec=True):
        ''' No training / optimization, this just tests the fed in dec '''
        
        if test_current_dec==True:
            test_dec = self.w
        else:
            #test_dec is whatever you input, presumably a matrix... probably should check
            test_dec = test_current_dec
            if np.prod(test_dec.shape)!=(self.PCA_comps*2):
                raise ValueError(f"Unexpected size of test_current_dec: {np.prod(test_dec.shape)} vs {self.PCA_comps*2} expected")
        
        # This sets FVD using the full client dataset
        # Since we aren't doing any optimization then it shouldn't matter if we use updates or not...
        simulate_data_stream(streaming_method='full_data')
        # Evaluate cost
        temp = cost_l2(self.F, test_dec, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)
        dec_cost = round(temp, 3)
        # Also want to see actual output 
        # This might be the cost and not the actual position...
        D_reshaped = np.reshape(test_dec,(2,self.PCA_comps))
        dec_pos = D_reshaped@self.F + self.H@self.V[:,:-1] - self.V[:,1:]
        return dec_cost, dec_pos


# Add this as a static method?
def condensed_external_plotting(input_data, version, exclusion_ID_lst=[], dim_reduc_factor=1, plot_gradient=False, plot_pers_gradient=False, plot_this_ID_only=-1, plot_global_gradient=False, global_error=True, local_error=True, pers_error=False, different_local_round_thresh_per_client=False, legend_on=False, plot_performance=False, plot_Dnorm=False, plot_Fnorm=False, num_participants=14, show_update_change=True, custom_title="", axes_off_list=[], ylim_max=None, ylim_min=None, my_legend_loc='best', global_alpha=1, local_alpha=1, pers_alpha=1, global_linewidth=1, local_linewidth=1, pers_linewidth=1, global_linestyle='dashed', local_linestyle='solid', pers_linestyle='dotted'):
    
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    
    def moving_average(numbers, window_size):
        i = 0
        moving_averages = []
        while i < len(numbers) - window_size + 1:
            this_window = numbers[i : i + window_size]

            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += window_size
        return moving_averages
    
    if custom_title:
        my_title = custom_title
    elif global_error and local_error:
        my_title = f'Global and Local Costs Per {version.title()} Iter'
    elif global_error:
        my_title = f'Global Cost Per {version.title()} Iter'
    elif local_error:
        my_title = f'Local Costs Per {version.title()} Iter'
    else:
        raise ValueError("You set both global and local to False.  At least one must be true in order to plot something.")

    # Determine if this is global or local, based on the input for now... could probably add a flag but meh
    if version.upper()=='LOCAL':
        user_database = input_data
    elif version.upper()=='GLOBAL':
        user_database = input_data.all_clients
    else:
        raise ValueError("log_type must be either global or local, please retry")
        
    max_local_iters = 0

    for i in range(len(user_database)):
        # Skip over users that distort the scale
        if user_database[i].ID in exclusion_ID_lst:
            continue 
        elif plot_this_ID_only!=-1 and i!=plot_this_ID_only:
            continue
        elif len(user_database[i].local_error_log)<2:
            # This node never trained so just skip it so it doesn't break the plotting
            continue 
        else: 
            # This is used for plotting later
            if len(user_database[i].local_error_log) > max_local_iters:
                max_local_iters = len(user_database[i].local_error_log)

            if version.upper()=='LOCAL':
                if global_error:
                    df = pd.DataFrame(user_database[i].global_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=global_linewidth, alpha=global_alpha, linestyle=global_linestyle)
                if local_error:
                    df = pd.DataFrame(user_database[i].local_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=local_linewidth, alpha=local_alpha, linestyle=local_linestyle)
                if pers_error:
                    df = pd.DataFrame(user_database[i].pers_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, alpha=pers_alpha, linestyle=pers_linestyle)
                # NOT THE COST FUNC, THESE ARE THE INDIVIDUAL COMPONENTS OF IT
                if plot_performance:
                    df = pd.DataFrame(user_database[i].performance_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, label=f"User{user_database[i].ID} Performance")
                if plot_Dnorm:
                    df = pd.DataFrame(user_database[i].Dnorm_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle="--", label=f"User{user_database[i].ID} Dnorm")
                if plot_Fnorm:
                    df = pd.DataFrame(user_database[i].Fnorm_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle=":", label=f"User{user_database[i].ID} Fnorm")
                if plot_gradient:
                    df = pd.DataFrame(user_database[i].gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Local Gradient")
                if plot_pers_gradient:
                    df = pd.DataFrame(user_database[i].pers_gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Pers Gradient")
                if plot_global_gradient:
                    df = pd.DataFrame(user_database[i].global_gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Global Gradient")
            elif version.upper()=='GLOBAL':
                if plot_Fnorm or plot_Dnorm or plot_performance:
                    print("Fnorm, Dnorm, and performance are currently not supported for version==GLOBAL")
                    
                if global_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.global_error_log[j][i][2])
                        # This is actually the client local round
                        client_global_round.append(input_data.global_error_log[j][i][1])
                    # Why is the [1:] here?  What happens when dim_reduc=1? 
                    # Verify that this is the same as my envelope code...
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=global_linewidth, alpha=global_alpha, linestyle=global_linestyle)

                if local_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.local_error_log[j][i][2])
                        client_global_round.append(input_data.local_error_log[j][i][1])
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=local_linewidth, alpha=local_alpha, linestyle=local_linestyle)
               
                if pers_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.pers_error_log[j][i][2])
                        client_global_round.append(input_data.pers_error_log[j][i][1])
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=pers_linewidth, alpha=pers_alpha, linestyle=pers_linestyle)

                if show_update_change:
                    for update_round in user_database[i].update_transition_log:
                        plt.axvline(x=(update_round), color=id2color[user_database[i].ID], linewidth=0.5, alpha=0.6)  

    if version.upper()=='LOCAL' and show_update_change==True:
        for i in range(max_local_iters):
            if i%user_database[0].local_round_threshold==0:
                plt.axvline(x=i, color="k", linewidth=1, linestyle=':')
                
    if axes_off_list!=[]:
        ax = plt.gca()
        for my_axis in axes_off_list:
            ax.spines[my_axis].set_visible(False)
              
    plt.ylabel('Cost L2')
    plt.xlabel('Iteration Number')
    plt.title(my_title)
    if version.upper()=='GLOBAL':
        max_local_iters = input_data.current_round
    else:
        num_ticks = 5
        plt.xticks(ticks=np.linspace(0,max_local_iters,num_ticks,dtype=int))
        plt.xlim((0,max_local_iters+1))
    if ylim_max!=None:
        if ylim_min!=None:
            plt.ylim((ylim_min,ylim_max))
        else:
            plt.ylim((0,ylim_max))
    if legend_on:
        plt.legend(loc=my_legend_loc)
    plt.show()
    

def central_tendency_plotting(all_user_input, highlight_default=False, default_local=False, default_global=False, default_pers=False, plot_mean=True, plot_median=False, exclusion_ID_lst=[], dim_reduc_factor=1, plot_gradient=False, plot_pers_gradient=False, plot_this_ID_only=-1, plot_global_gradient=False, global_error=True, local_error=True, pers_error=False, different_local_round_thresh_per_client=False, legend_on=True, plot_performance=False, plot_Dnorm=False, plot_Fnorm=False, num_participants=14, show_update_change=True, custom_title="", axes_off_list=[], ylim_max=None, ylim_min=None, input_linewidth=1, my_legend_loc='best', iterable_labels=[], iterable_colors=[]):
    
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    
    num_central_tendencies = 2  # Mean and median... idk, maybe use flags or something...
    
    if dim_reduc_factor!=1:
        raise("dim_reduc_factor MUST EQUAL 1!")
        
    global_df = pd.DataFrame()
    local_df = pd.DataFrame()
    pers_df = pd.DataFrame()
    perf_df = pd.DataFrame()
    dnorm_df = pd.DataFrame()
    fnorm_df = pd.DataFrame()
    grad_df = pd.DataFrame()
    pers_grad_df = pd.DataFrame()
    global_grad_df = pd.DataFrame()
    
    param_list = [plot_gradient, plot_pers_gradient, plot_global_gradient, global_error, local_error, pers_error, plot_performance, plot_Dnorm, plot_Fnorm]
    all_vecs_dict = dict()
    all_vecX_dict = dict()
    for param_idx, param in enumerate(param_list):
        all_vecs_dict[param_idx] = [[] for _ in range(num_central_tendencies)]
        all_vecX_dict[param_idx] = [[] for _ in range(num_central_tendencies)]
    param_label_dict = {0:'Gradient', 1:'Personalized Gradient', 2:'Global Gradient', 3:'Global Error', 4:'Local Error', 5:'Personalized Error', 6:'Performance', 7:'DNorm', 8:'FNorm'}
    tendency_label_dict = {0:'Mean', 1:'Pseudo-Median'}
    
    def moving_average(numbers, window_size):
        i = 0
        moving_averages = []
        while i < len(numbers) - window_size + 1:
            this_window = numbers[i : i + window_size]

            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += window_size
        return moving_averages
    
    if custom_title:
        my_title = custom_title
    elif global_error and local_error:
        my_title = f'Global and Local Costs Per Local Iter'
    elif global_error:
        my_title = f'Global Cost Per Local Iter'
    elif local_error:
        my_title = f'Local Costs Per Local Iter'
    else:
        raise ValueError("You set both global and local to False.  At least one must be true in order to plot something.")

    max_local_iters = 0
    
    for user_idx, user_database in enumerate(all_user_input):
        for i in range(len(user_database)):
            # Skip over users that distort the scale
            if user_database[i].ID in exclusion_ID_lst:
                continue 
            elif len(user_database[i].local_error_log)<2:
                # This node never trained so just skip it so it doesn't break the plotting
                continue 
            else: 
                # This is used for plotting later
                if len(user_database[i].local_error_log) > max_local_iters:
                    max_local_iters = len(user_database[i].local_error_log)

                # This is how it would be supposed to work
                # Append needs to change to concat
                # ISSUE: I loop through the iters and so the dfs aren't actually built yet
                # So I would have to build each df (do concat and have some base init...)
                #for flag_idx, plotting_flag in enumerate(param_list):
                #    if plotting_flag:
                #        df = pd.DataFrame(user_database[i].global_error_log)
                #         df.reset_index(inplace=True)
                #        global_df.append(df.groupby(df.index//dim_reduc_factor, axis=0).mean())
                #        all_dfs_dict[flag_idx]
                #        for column in my_df:
                #            if 'MEAN' in central_tendency.upper():
                    #            all_vecs_dict[flag_idx].append(pd.DataFrame(my_df[column].tolist().mean().tolist()))
                #            if 'MEDIAN' in central_tendency.upper():
                #                all_vecs_dict[flag_idx].append(pd.DataFrame(my_df.median(axis=0)))
                if global_error or (user_idx==0 and default_global==True):
                    df = pd.DataFrame(user_database[i].global_error_log)
                    global_df = pd.concat([global_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if local_error or (user_idx==0 and default_local==True):
                    df = pd.DataFrame(user_database[i].local_error_log)
                    local_df = pd.concat([local_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if pers_error or (user_idx==0 and default_pers==True):
                    df = pd.DataFrame(user_database[i].pers_error_log)
                    pers_df = pd.concat([pers_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_performance:
                    df = pd.DataFrame(user_database[i].performance_log)
                    perf_df = pd.concat([perf_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_Dnorm:
                    df = pd.DataFrame(user_database[i].Dnorm_log)
                    dnorm_df = pd.concat([dnorm_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_Fnorm:
                    df = pd.DataFrame(user_database[i].Fnorm_log)
                    fnorm_df = pd.concat([fnorm_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_gradient:
                    df = pd.DataFrame(user_database[i].gradient_log)
                    grad_df = pd.concat([grad_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_pers_gradient:
                    df = pd.DataFrame(user_database[i].pers_gradient_log)
                    pers_grad_df = pd.concat([pers_grad_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_global_gradient:
                    df = pd.DataFrame(user_database[i].global_gradient_log)
                    global_grad_df = pd.concat([global_grad_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])

        # Bad temporary soln for MVP
        all_dfs_dict = {0:grad_df.reset_index(drop=True), 1:pers_grad_df.reset_index(drop=True), 2:global_grad_df.reset_index(drop=True), 3:global_df.reset_index(drop=True), 4:local_df.reset_index(drop=True), 5:pers_df.reset_index(drop=True), 6:perf_df.reset_index(drop=True), 7:dnorm_df.reset_index(drop=True), 8:fnorm_df.reset_index(drop=True)}
        
        for flag_idx, plotting_flag in enumerate(param_list):
            if plotting_flag:
                my_df = all_dfs_dict[flag_idx]
                if plot_mean:
                    all_vecs_dict[flag_idx][0] = my_df.mean()
                if plot_median:
                    all_vecs_dict[flag_idx][1] = my_df.median()

        if show_update_change==True:
            for i in range(max_local_iters):
                if i%user_database[0].local_round_threshold==0:
                    plt.axvline(x=i, color="k", linewidth=1, linestyle=':') 

        for flag_idx, plotting_flag in enumerate(param_list):
            if plotting_flag:
                my_vec = all_vecs_dict[flag_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        if iterable_labels!=[]:
                            my_label = iterable_labels[user_idx]
                        else:
                            my_label = f"{tendency_label_dict[vec_idx]} {param_label_dict[flag_idx]}"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
                        
                        
        #param_list FOR REFERENCE: [plot_gradient, plot_pers_gradient, plot_global_gradient, global_error, local_error, pers_error, plot_performance, plot_Dnorm, plot_Fnorm]         
        if user_idx==0:
            if default_global:  # 3 corresponds to global
                global_idx = 3
                all_vecs_dict[global_idx][0] = all_dfs_dict[global_idx].mean()
                all_vecs_dict[global_idx][1] = all_dfs_dict[global_idx].median()
                my_vec = all_vecs_dict[global_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        my_label = f"{tendency_label_dict[vec_idx]} Global Error"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
            if default_local:  # 4 corresponds to local
                local_idx = 4
                all_vecs_dict[local_idx][0] = all_dfs_dict[local_idx].mean()
                all_vecs_dict[local_idx][1] = all_dfs_dict[local_idx].median()
                my_vec = all_vecs_dict[local_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        my_label = f"{tendency_label_dict[vec_idx]} Local Error"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
            if default_pers:  # 5 corresponds to pers
                pers_idx = 5
                all_vecs_dict[pers_idx][0] = all_dfs_dict[pers_idx].mean()
                all_vecs_dict[pers_idx][1] = all_dfs_dict[pers_idx].median()
                my_vec = all_vecs_dict[pers_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        my_label = f"{tendency_label_dict[vec_idx]} Personalized Error"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
    
    plt.ylabel('Cost L2')
    plt.xlabel('Iteration Number')
    plt.title(my_title)
    num_ticks = 5
    plt.xticks(ticks=np.linspace(0,max_local_iters,num_ticks,dtype=int))
    plt.xlim((0,max_local_iters+1))
    if ylim_max!=None:
        if ylim_min!=None:
            plt.ylim((ylim_min,ylim_max))
        else:
            plt.ylim((0,ylim_max))
    if legend_on:
        plt.legend(loc=my_legend_loc)
    
    if axes_off_list!=[]:
        ax = plt.gca()
        for my_axis in axes_off_list:
            ax.spines[my_axis].set_visible(False)
        
    plt.show()
    
    return user_database, all_dfs_dict, all_vecs_dict
    
    
##############################################################################
# Zero vel boundary code:
def reconstruct_trial_fixed_decoder(ref_tr, emg_tr, Ds_fixed, time_x, fs = 60):
    time_x = time_x
    vel_est = np.zeros_like((ref_tr))
    pos_est = np.zeros_like((ref_tr))
    int_vel_est = np.zeros_like((ref_tr))

    hit_bound = 0
    vel_est[0] = Ds_fixed@emg_tr[0]  # D@s --> Kai's v_actual
    pos_est[0] = [0, 0]
    for tt in range(1, time_x):
        vel_plus = Ds_fixed@emg_tr[tt] # at time tt --> also Kai's v_actual...
        p_plus = pos_est[tt-1, :] + (vel_est[tt-1, :]/fs)
        # These are just correctives, such that vel_plus can get bounded
        # x-coordinate
        if abs(p_plus[0]) > 36:
            p_plus[0] = pos_est[tt-1, 0]
            vel_plus[0] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter
        if abs(p_plus[1]) > 24:
            p_plus[1] = pos_est[tt-1, 1]
            vel_plus[1] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter
        if hit_bound > 200:
            p_plus[0] = 0
            vel_plus[0] = 0
            p_plus[1] = 0
            vel_plus[1] = 0
            hit_bound = 0
        # now update velocity and position
        vel_est[tt] = vel_plus
        pos_est[tt] = p_plus
        # calculate intended velocity
        int_vel_est[tt] = calculate_intended_vels(ref_tr[tt], p_plus, 60)
    return vel_est, pos_est, int_vel_est


def calculate_intended_vels(ref, pos, fs):
    '''
    ref = 1 x 2
    pos = 1 x 2
    fs = scalar
    '''
    
    gain = 120
    ALMOST_ZERO_TOL = 0.01
    intended_vector = (ref - pos)/fs
    if np.linalg.norm(intended_vector) <= ALMOST_ZERO_TOL:
        intended_norm = np.zeros((2,))
    else:
        intended_norm = intended_vector * gain
    return intended_norm