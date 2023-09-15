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


class Client(ModelBase, TrainingMethods):
    def __init__(self, ID, w, method, local_data, data_stream, smoothbatch=1, current_round=0, PCA_comps=7, availability=1, global_method='FedAvg', normalize_dec=False, normalize_EMG=True, starting_update=0, track_cost_components=True, track_lr_comps=True, use_real_hess=True, gradient_clipping=False, log_decs=True, clipping_threshold=100, tol=1e-10, adaptive=True, eta=1, track_gradient=True, wprev_global=False, num_steps=1, use_zvel=False, APFL_input_eta=False, safe_lr_factor=False, set_alphaF_zero=False, mix_in_each_steps=False, mix_mixed_SB=False, delay_scaling=5, random_delays=False, download_delay=1, upload_delay=1, copy_type='deep', validate_memory_IDs=True, local_round_threshold=25, condition_number=1, verbose=False, test_split_type='end', test_split_frac=0.3, use_up16_for_test=True):
        super().__init__(ID, w, method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, verbose=verbose, num_participants=14, log_init=0)
        '''
        Note self.smoothbatch gets overwritten according to the condition number!  
        If you want NO smoothbatch then set it to 'off'
        '''
        self.validate_memory_IDs = validate_memory_IDs
        self.copy_type = copy_type
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
        # TRAIN TEST DATA SPLIT
        self.test_split_type = test_split_type
        self.test_split_frac = test_split_frac
        self.use_up16_for_test = use_up16_for_test
        test_split_product_index = local_data['training'].shape[0]*test_split_frac
        # Convert this value to the cloest update_ix value
        train_test_update_number_split = min(ModelBase.update_ix, key=lambda x:abs(x-test_split_product_index))
        print(f"use_up16_for_test: {self.use_up16_for_test}")
        print(f"test_split_product_index = local_data['training'].shape[0]*test_split_frac = {test_split_product_index}")
        print(f"train_test_update_number_split (closest update): {train_test_update_number_split}")
        self.test_split_idx = ModelBase.update_ix[train_test_update_number_split]
        print(f"test_split_product_index (actual data index): {self.test_split_idx}")
        self.training_data = local_data['training']#[:self.test_split_idx, :]
        self.labels = local_data['labels']#[:self.test_split_idx, :]
        #self.testing_data = local_data['training'][self.test_split_idx:, :]
        #self.testing_labels = local_data['labels'][self.test_split_idx:, :]
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
        if set_alphaF_zero:
            self.alphaF = 0
        else:
            self.alphaF = 1e-7
        #
        self.gradient_clipping = gradient_clipping
        self.clipping_threshold = clipping_threshold
        # PLOTTING
        self.log_decs = log_decs
        self.pers_dec_log = [np.zeros((2,self.PCA_comps))]
        self.global_dec_log = [np.zeros((2,self.PCA_comps))]
        # Overwrite the logs since global and local track in slightly different ways
        # TRAINING LOSS
        self.local_error_log = []
        self.global_error_log = []
        self.pers_error_log = []
        # TESTING LOSS
        self.local_test_error_log = []
        self.global_test_error_log = []
        self.pers_test_error_log = []
        #
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
        
        # SET TESTING DATA AND METRICS
        if self.use_up16_for_test==True:
            lower_bound = (ModelBase.update_ix[15])//2  #Use only the second half of each update
            upper_bound = ModelBase.update_ix[16]
        else:
            raise ValueError("use_up16_for_test must be True, it is the only testing supported currently")
        self.test_learning_batch = upper_bound - lower_bound
        # THIS NEEDS TO BE FIXED...
        s_temp = self.training_data[lower_bound:upper_bound,:]
        if self.normalize_EMG:
            s_normed = s_temp/np.amax(s_temp)
        else:
            s_normed = s_temp
        if self.PCA_comps!=self.pca_channel_default:  
            pca = PCA(n_components=self.PCA_comps)
            s_normed = pca.fit_transform(s_normed)
        s = np.transpose(s_normed)
        self.F_test = s[:,:-1] # note: truncate F for estimate_decoder
        v_actual = model@s
        p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
        p_reference = np.transpose(self.labels[lower_bound:upper_bound,:])
        self.V_test = (p_reference - p_actual)*self.dt
            
            
    # 0: Main Loop
    def execute_training_loop(self):
        self.simulate_data_stream()
        self.train_model()
        
        # LOG EVERYTHING
        # Log decs
        if self.log_decs:
            self.dec_log.append(self.w)
            if self.global_method=="FedAvg":
                self.global_dec_log.append(copy.deepcopy(self.global_w))
            elif self.global_method in self.pers_methods:
                self.global_dec_log.append(copy.deepcopy(self.global_w))
                self.pers_dec_log.append(copy.deepcopy(self.mixed_w))
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
            lower_bound = (ModelBase.update_ix[-3] + ModelBase.update_ix[-2])//2  #Use only the second half of each update
            upper_bound = ModelBase.update_ix[-2]
            self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='full_data':
            lower_bound = ModelBase.update_ix[0]  # Starts at 0 and not update 10, for now
            upper_bound = ModelBase.update_ix[-1]
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
                lower_bound = (ModelBase.update_ix[self.current_update] + ModelBase.update_ix[self.current_update+1])//2  
                upper_bound = ModelBase.update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
            elif self.current_round>2:
                # This is the base case
                # The update number didn't change so we don't need to overwrite everything with the same data
                need_to_advance = False
            else:
                # This is for the init case (current round is 0 or 1)
                # need_to_advance is true, so we overwrite s and such... this is fine 
                lower_bound = (ModelBase.update_ix[self.current_update] + ModelBase.update_ix[self.current_update+1])//2  
                upper_bound = ModelBase.update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='advance_each_iter':
            lower_bound = (ModelBase.update_ix[self.current_update] + ModelBase.update_ix[self.current_update+1])//2  
            upper_bound = ModelBase.update_ix[self.current_update+1]
            self.learning_batch = upper_bound - lower_bound
            
            self.current_update += 1
        else:
            raise ValueError(f'streaming_method ("{streaming_method}") not recognized: this data streaming functionality is not supported')
            
        if need_to_advance:
            s_temp = self.training_data[lower_bound:upper_bound,:]
            ###########################################################################################################################
            # First, normalize the entire s matrix
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            ###########################################################################################################################
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
    

    def train_given_model_1_comm_round(self, model, which):
        '''This can be used for training the in ML pipeline but principally is for local-finetuning (eg training a model after it as completed its global training pipeline).'''
        
        D_0 = copy.deepcopy(self.w_prev)
        # Set the w_prev equal to the current w:
        self.w_prev = copy.deepcopy(model)
        if self.global_method in ["FedAvg", "NoFL", "FedAvgSB", "Per-FedAvg", "Per-FedAvg FO", "Per-FedAvg HF"]:
            for i in range(self.num_steps):
                if self.normalize_dec:
                    model /= np.amax(model)
                    
                if self.method=='EtaGradStep':
                    model = self.train_eta_gradstep(model, self.eta, self.F, model, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, PCA_comps=self.PCA_comps)
                elif self.method=='EtaScipyMinStep':
                    model = self.train_eta_scipyminstep(self.w, self.eta, self.F, model, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, D_0, self.verbose, PCA_comps=self.PCA_comps)
                elif self.method=='FullScipyMinStep':
                    model = self.train_eta_scipyminstep(model, self.eta, self.F, model, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, D_0, self.verbose, PCA_comps=self.PCA_comps, full=True)
                else:
                    raise ValueError("Unsupported method")
                    
                #if self.mix_in_each_steps:
                #    raise("mix_in_each_steps=True: Functionality not yet supported")
                #    self.mixed_w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.mixed_w)
            
            if self.normalize_dec:
                model /= np.amax(model)
            
            # Do SmoothBatch
            #if self.global_method in ["FedAvg", "NoFL"]:  # Maybe should add Per-FedAvg here...
            #    model = self.smoothbatch*model + ((1 - self.smoothbatch)*self.w_prev)
            #elif self.global_method=="FedAvgSB":
            #    global_local_SB = self.smoothbatch*model + ((1 - self.smoothbatch)*self.global_w)
            #    if self.mix_mixed_SB:
            #        self.mixed_w = self.smoothbatch*self.mixed_w + ((1 - self.smoothbatch)*global_local_SB)
            #    else:
            #        self.mixed_w = global_local_SB
        
        if which!=None:
            return model, self.eval_model(which)
        else:
            return model
    
    
    def train_model(self):
        D_0 = copy.deepcopy(self.w_prev)
        # Set the w_prev equal to the current w:
        self.w_prev = copy.deepcopy(self.w)
        if self.global_method in ["FedAvg", "NoFL", "FedAvgSB", "Per-FedAvg", "Per-FedAvg FO", "Per-FedAvg HF"]:
            if self.global_method!="NoFL":
                # Overwrite local model with the new global model
                if self.copy_type == 'deep':
                    self.w = copy.deepcopy(self.global_w)
                elif self.copy_type == 'shallow':
                    self.w = copy.copy(self.global_w)
                elif self.copy_type == 'none':
                    self.w = self.global_w
                else:
                    raise ValueError("copy_type must be set to either deep, shallow, or none")
            
            # I think this ought to be on but it makes the global model and gradient diverge...
            if self.wprev_global==True and ('Per-FedAvg' in self.method):
                if self.copy_type == 'deep':
                    self.w_prev = copy.deepcopy(self.global_w)
                elif self.copy_type == 'shallow':
                    self.w_prev = copy.copy(self.global_w)
                elif self.copy_type == 'none':
                    self.w_prev = self.global_w
                else:
                    raise ValueError("copy_type must be set to either deep, shallow, or none")
            
            for i in range(self.num_steps):
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
                    delta = self.eta  # Not sure... value not listed in paper
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
                
                if self.validate_memory_IDs:
                    assert(id(self.w)!=id(self.global_w))
                    assert(id(self.w_prev)!=id(self.global_w))
                    
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
            out = round(temp, 7)
        return out
        
    def test_metrics(self, model, which, final_eval=False):
        ''' No training / optimization, this just tests the fed in dec '''
        
        if which=='local':
            test_log = self.local_test_error_log
        elif which=='global':
            test_log = self.global_test_error_log
        elif which=='pers':
            test_log = self.pers_test_error_log
        else:
            raise ValueError('test_metrics which does not exist. Must be local, global, or pers')
            
        if self.use_up16_for_test==True and final_eval==True:
            lower_bound = (ModelBase.update_ix[16])//2  #Use only the second half of each update
            upper_bound = ModelBase.update_ix[17]
            
            self.test_learning_batch = upper_bound - lower_bound
            # THIS NEEDS TO BE FIXED...
            s_temp = self.training_data[lower_bound:upper_bound,:]
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            if self.PCA_comps!=self.pca_channel_default:  
                pca = PCA(n_components=self.PCA_comps)
                s_normed = pca.fit_transform(s_normed)
            s = np.transpose(s_normed)
            self.F_test = s[:,:-1] # note: truncate F for estimate_decoder
            v_actual = model@s
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            p_reference = np.transpose(self.labels[lower_bound:upper_bound,:])
            self.V_test = (p_reference - p_actual)*self.dt
        elif self.use_up16_for_test==True and final_eval==False:
            # Can reuse the test values already set earlier in the init func
            pass
        else:
            raise ValueError("use_up16_for_test must be True, it is the only testing supported currently")
        
        test_loss = cost_l2(self.F_test, model, self.H, self.V_test, self.test_learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)
        self.test_log.append(test_loss)
        
        # OLD VERSION THAT WAS NEVER USED
        '''
        if test_current_dec==True:
            test_dec = self.w
        else:
            #test_dec is whatever you input, presumably a matrix... probably should check
            test_dec = test_current_dec
            if np.prod(test_dec.shape)!=(self.PCA_comps*2):
                raise ValueError(f"Unexpected size of test_current_dec: {np.prod(test_dec.shape)} vs {self.PCA_comps*2} expected")
        
        # This sets FVD using the full client dataset
        # ^... testing on the full dataset (especially all at once) seems stupid...
        # Since we aren't doing any optimization then it shouldn't matter if we use updates or not...
        simulate_data_stream(streaming_method='full_data')
        
        # Evaluate cost
        temp = cost_l2(self.F, test_dec, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)
        dec_cost = round(temp, 7)
        
        # Also want to see actual output AKA predicted output (I forget if this is position or velocity)
        # This might be the cost and not the actual position...
        D_reshaped = np.reshape(test_dec,(2,self.PCA_comps))
        dec_pos = D_reshaped@self.F + self.H@self.V[:,:-1] - self.V[:,1:]
        '''
        return dec_cost, dec_pos