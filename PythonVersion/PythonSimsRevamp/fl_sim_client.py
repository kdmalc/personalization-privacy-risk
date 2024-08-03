import numpy as np
import random
import copy

from experiment_params import *
from cost_funcs import *
import time
from sklearn.decomposition import PCA

from scipy.optimize import minimize

from fl_sim_base import *


class Client(ModelBase):
    def __init__(self, ID, w, opt_method, full_client_dataset, data_stream, smoothbatch=0.75, current_round=0, PCA_comps=64, 
                availability=1, final_usable_update_ix=17, global_method='FedAvg', max_iter=1, normalize_EMG=True, starting_update=10, 
                track_cost_components=True, gradient_clipping=False, log_decs=True, 
                clipping_threshold=100, tol=1e-10, lr=1, track_gradient=True, wprev_global=False, 
                num_steps=1, use_zvel=False, use_kfoldv=False, 
                mix_in_each_steps=False, mix_mixed_SB=False, delay_scaling=0, random_delays=False, download_delay=1, 
                upload_delay=1, validate_memory_IDs=True, local_round_threshold=50, condition_number=3, 
                verbose=False, test_split_type='end', test_split_frac=0.3, use_up16_for_test=False):
        super().__init__(ID, w, opt_method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, 
                         verbose=verbose, num_participants=14, log_init=0)
        '''
        Note self.smoothbatch gets overwritten according to the condition number!  
        If you want NO smoothbatch then set it to 'off'
        '''

        assert(full_client_dataset['training'].shape[1]==64) # --> Shape is (20770, 64)
        # Don't use anything past update 17 since they are different (update 17 is the short one, only like 300 datapoints)
        self.local_training_data = full_client_dataset['training'][:self.update_ix[final_usable_update_ix], :]
        self.local_training_labels = full_client_dataset['labels'][:self.update_ix[final_usable_update_ix], :]

        self.global_method = global_method.upper()
        self.validate_memory_IDs = validate_memory_IDs
        # NOT INPUT
        self.type = 'Client'
        self.chosen_status = 0
        self.latest_global_round = 0
        self.update_transition_log = []
        self.normalize_EMG = normalize_EMG
        # Sentinel Values
        self.F = None
        self.V = None
        self.learning_batch = None

        self.dt = 1.0/60.0
        self.lr = lr  # Learning rate
        # Round minimization output to the nearest int or keep as a float?  Don't need arbitrary precision
        self.round2int = False
        self.max_iter = max_iter
        
        # Maneeshika Code:
        self.use_zvel = use_zvel
        self.hit_bound = 0
        
        # FL CLASS STUFF
        # Availability for training
        self.availability = availability
        # Toggle streaming aspect of data collection: {Ignore updates and use all the data; 
        #  Stream each update, moving to the next update after local_round_threshold iters have been run; 
        #  After 1 iteration, move to the next update}
        self.data_stream = data_stream  # {'full_data', 'streaming', 'advance_each_iter'} 
        # Number of gradient steps to take when training (eg amount of local computation)
        ## This is Tau in PFA!
        self.num_steps = num_steps
        self.wprev_global = wprev_global # Not sure what this is honestly
        # GLOBAL STUFF
        self.global_method = global_method
        # UPDATE STUFF
        self.current_update = starting_update
        self.local_round_threshold = local_round_threshold

        # PRACTICAL / REAL WORLD
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
            #print()
        self.alphaE = 1e-6
        self.alphaF = 0
        # This is probably not used...
        self.gradient_clipping = gradient_clipping
        self.clipping_threshold = clipping_threshold
        # LOGS
        self.log_decs = log_decs
        self.pers_dec_log = [np.zeros((2,self.PCA_comps))]
        self.global_dec_log = [np.zeros((2,self.PCA_comps))]
        # Overwrite the logs since global and local track in slightly different ways
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

        # These are general personalization things
        ## Not sure if any of these are even used...
        self.running_pers_term = 0
        self.running_global_term = 0
        self.global_w = copy.deepcopy(self.w)
        self.mixed_w = copy.deepcopy(self.w)

        # TRAIN TEST DATA SPLIT
        self.use_up16_for_test = use_up16_for_test
        self.use_kfoldcv = use_kfoldv
        self.test_split_type = test_split_type.upper()
        self.test_split_frac = test_split_frac

        if self.use_kfoldcv==True:
            self.test_split_idx = -1
            # Setting the testing set to the whole dataset so that it can be extracted
            ## If this is a training cliet this will be overwritten by other client's testing dataset
            ## I guess I could set this to 
            upper_bound = self.local_training_data.shape[0]
            lower_bound = 0
            self.testing_data = self.local_training_data
            self.testing_labels = self.local_training_labels
        elif self.use_up16_for_test==True:
            lower_bound = self.update_ix[15]
            #//2  #Use only the second half of each update
            upper_bound = self.update_ix[16]
            self.test_split_idx = lower_bound
            self.testing_data = self.local_training_data[self.test_split_idx:upper_bound, :]
            self.testing_labels = self.local_training_labels[self.test_split_idx:upper_bound, :]
            # TODO: There ought to be some assert here to make sure that self.testing_XYZ doesnt have a shape of zero...
        elif self.test_split_type=="END":
            test_split_product_index = self.local_training_data.shape[0]*test_split_frac
            # Convert this value to the cloest update_ix value
            train_test_update_number_split = min(self.update_ix, key=lambda x:abs(x-test_split_product_index))
            if train_test_update_number_split<17:
                print(f"train_test_update_number_split is {train_test_update_number_split}, so subtracting 1")
                train_test_update_number_split -= 1
            self.test_split_idx = self.update_ix.index(train_test_update_number_split)
            lower_bound = self.test_split_idx
            upper_bound = self.local_training_data.shape[0]
            self.testing_data = self.local_training_data[self.test_split_idx:, :]
            self.testing_labels = self.local_training_labels[self.test_split_idx:, :]
        else:
            raise ValueError("use_up16_for_test or use_kfoldcv must be True or test_split_type must be END, they are the only 3 testing schemes supported currently")
        
        self.training_data = self.local_training_data[:self.test_split_idx, :]
        self.labels = self.local_training_labels[:self.test_split_idx, :]
        self.test_learning_batch = upper_bound - lower_bound
        s_temp = self.testing_data
        if self.normalize_EMG:
            s_normed = s_temp/np.amax(s_temp)
        else:
            s_normed = s_temp
        if self.PCA_comps!=self.pca_channel_default:  
            pca = PCA(n_components=self.PCA_comps)
            s_normed = pca.fit_transform(s_normed)
        self.s_test = np.transpose(s_normed)
        self.F_test = self.s_test[:,:-1] # note: truncate F for estimate_decoder
        self.p_test_reference = np.transpose(self.testing_labels)
        
            
    # 0: Main Loop
    def execute_training_loop(self):
        self.simulate_data_stream()
        self.train_model()
        
        # LOG EVERYTHING
        # Log decs
        if self.log_decs:
            self.dec_log.append(self.w)
            if self.global_method=="FEDAVG":
                self.global_dec_log.append(copy.deepcopy(self.global_w))
            elif self.global_method in self.pers_methods:
                self.global_dec_log.append(copy.deepcopy(self.global_w))
                self.pers_dec_log.append(copy.deepcopy(self.mixed_w))
        # Log Error
        self.local_error_log.append(self.eval_model(which='local'))
        # Yes these should both be ifs not elif, they may both need to run
        if self.global_method!="NOFL":
            self.global_error_log.append(self.eval_model(which='global'))
        if self.global_method in self.pers_methods:
            self.pers_error_log.append(self.eval_model(which='pers'))
        # Log Cost Comp
        if self.track_cost_components:
            self.performance_log.append(self.alphaE*(np.linalg.norm((self.w@self.F - self.V[:,1:]))**2))
            self.Dnorm_log.append(self.alphaD*(np.linalg.norm(self.w)**2))
            self.Fnorm_log.append(self.alphaF*(np.linalg.norm(self.F)**2))


    def set_testset(self, test_dataset_obj):
        lower_bound = 0

        if type(test_dataset_obj) is type([]):
            running_num_test_samples = 0
            self.s_test = []
            self.F_test = []
            self.p_test_reference = []
            for test_dset in test_dataset_obj:
                self.testing_data = test_dset[0]
                self.testing_labels = test_dset[1]

                # TODO: Verify that shape index is correct
                running_num_test_samples += self.testing_data.shape[0]

                s_temp = self.testing_data
                if self.normalize_EMG:
                    s_normed = s_temp/np.amax(s_temp)
                else:
                    s_normed = s_temp
                if self.PCA_comps!=self.pca_channel_default:  
                    pca = PCA(n_components=self.PCA_comps)
                    s_normed = pca.fit_transform(s_normed)
                self.s_test.append(np.transpose(s_normed))
                self.F_test.append(np.transpose(s_normed)[:,:-1]) # note: truncate F for estimate_decoder
                self.p_test_reference.append(np.transpose(self.testing_labels))

            upper_bound = running_num_test_samples
            # TODO: This might be incorrect... it is calculated differently than the below branch...
            ## Eg maybe this should be a list of each upper-lower? Not sure...
            ## Probably doesn't matter? Not even sure if this is used
            self.test_learning_batch = upper_bound - lower_bound
        else:
            self.testing_data = test_dataset_obj[0]
            self.testing_labels = test_dataset_obj[1]
            # TODO: Verify that shape index is correct
            upper_bound = self.testing_data.shape[0]

            self.test_learning_batch = upper_bound - lower_bound

            s_temp = self.testing_data
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            if self.PCA_comps!=self.pca_channel_default:  
                pca = PCA(n_components=self.PCA_comps)
                s_normed = pca.fit_transform(s_normed)
            self.s_test = np.transpose(s_normed)
            self.F_test = self.s_test[:,:-1] # note: truncate F for estimate_decoder
            self.p_test_reference = np.transpose(self.testing_labels)
    
    
    def get_testing_dataset(self):
        return self.testing_data, self.testing_labels

        
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
            lower_bound = (self.update_ix[-3] + self.update_ix[-2])//2  #Use only the second half of each update
            upper_bound = self.update_ix[-2]
            self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='full_data':
            lower_bound = self.update_ix[0]  # Starts at 0 and not update 10, for now
            upper_bound = self.update_ix[-1]
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
                lower_bound = (self.update_ix[self.current_update] + self.update_ix[self.current_update+1])//2  
                upper_bound = self.update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
            elif self.current_round>2:
                # This is the base case
                # The update number didn't change so we don't need to overwrite everything with the same data
                need_to_advance = False
            else:
                # This is for the init case (current round is 0 or 1)
                # need_to_advance is true, so we overwrite s and such... this is fine 
                lower_bound = (self.update_ix[self.current_update] + self.update_ix[self.current_update+1])//2  
                upper_bound = self.update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='advance_each_iter':
            lower_bound = (self.update_ix[self.current_update] + self.update_ix[self.current_update+1])//2  
            upper_bound = self.update_ix[self.current_update+1]
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
            

    def train_given_model_1_comm_round(self, model, which):
        '''This can be used for training the in ML pipeline but principally is for local-finetuning (eg training a model after it as completed its global training pipeline).'''
        raise ValueError("train_given_model_1_comm_round is not implemented yet.")
    
    
    def train_model(self):
        ## Set the w_prev equal to the current w:
        self.w_prev = copy.deepcopy(self.w)

        if self.global_method!="NOFL":
            # Overwrite local model with the new global model
            # Is this the equivalent of recieving the global model I'm assuming?
            ## So... this should be D0 and such right...
            self.w_new = copy.deepcopy(self.global_w)
        else:
            # w_new is just the same model it has been
            self.w_new = copy.deepcopy(self.w)
        
        # I think this ought to be on but it makes the global model and gradient diverge...
        if self.wprev_global==True and ('PFA' in self.opt_method):
            # TODO: What is wprev_global...
            self.w_prev = copy.deepcopy(self.global_w)
        
        for i in range(self.num_steps):
            D0 = copy.deepcopy(self.w_new)  
            # ^ Was self.w_prev for some reason...
            ## But it should be the current model at the start

            if self.opt_method=='GD':
                grad_cost = np.reshape(gradient_cost_l2(self.F, self.w_new, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                                        (2, self.PCA_comps))
                self.w_new -= self.lr*grad_cost
            elif self.opt_method=='FULLSCIPYMIN':
                out = minimize(
                    lambda D: cost_l2(self.F, D, self.V, alphaD=self.alphaD, alphaE=self.alphaE, Ne=self.PCA_comps), 
                    D0, method='BFGS', 
                    jac=lambda D: gradient_cost_l2(self.F, D, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps))
                self.w_new = np.reshape(out.x,(2, self.PCA_comps))
            elif self.opt_method=='MAXITERSCIPYMIN':
                out = minimize(
                    lambda D: cost_l2(self.F, D, self.V, alphaD=self.alphaD, alphaE=self.alphaE, Ne=self.PCA_comps), 
                    D0, method='BFGS', 
                    jac=lambda D: gradient_cost_l2(self.F, D, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                    options={'maxiter':self.max_iter})
                self.w_new = np.reshape(out.x,(2, self.PCA_comps))
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            elif self.opt_method=='PFAFO':
                stochastic_grad = np.reshape(gradient_cost_l2(self.F, D0, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                                        (2, self.PCA_comps))
                w_tilde = D0 - self.lr * stochastic_grad
                # ^ D0 is w_new from the previous iteration, eg w_{t-1}
                # TODO: Decide what to do about F and V... split in half? Use batches? ...
                new_stoch_grad = np.reshape(gradient_cost_l2(self.F, w_tilde, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                                        (2, self.PCA_comps))
                self.w_new = D0 - self.beta * new_stoch_grad
            elif self.opt_method=='PFAHF':
                # TODO: Implement. Is the hessian-vector product between the hessian and the gradient? ...
                raise ValueError('Per-FedAvg HF NOT FINISHED YET')
            else:
                raise ValueError("Unrecognized method")
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################
            ##################################################################################

            if self.validate_memory_IDs:
                # TODO: Idek what this is doing. This is stupid to have
                assert(id(self.w)!=id(self.global_w))
                assert(id(self.w_prev)!=id(self.global_w))
                
            if self.mix_in_each_steps:
                # TODO: mixed_w appears to not be used // is used as the personalized model?...
                self.mixed_w = self.smoothbatch*self.w_new + ((1 - self.smoothbatch)*self.mixed_w)

        # Do SmoothBatch
        #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
        # TODO: Add a smoothbatch toggle here
        self.w = self.smoothbatch*self.w_new + ((1 - self.smoothbatch)*self.w_prev)

        # TODO: What is this doing here? Does this get logged elsewhere or what?
        # Save the new decoder to the log
        #self.dec_log.append(self.w)
        #if self.global_method in self.pers_methods:
        #    self.pers_dec_log.append(self.mixed_w)
        #self.global_dec_log.append(self.global_w)
        
        
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
            temp = np.ceil(cost_l2(self.F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps))
            # Setting to int is just to catch overflow errors
            # For RT considerations, ints are also generally ints cheaper than floats...
            out = int(temp)
        else:
            temp = cost_l2(self.F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
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
            
        # Idk what this is doing at all... wasn't this already done in the init func...
        #if self.use_up16_for_test==True and final_eval==True:
        #    lower_bound = (self.update_ix[16])//2  #Use only the second half of each update
        #    upper_bound = self.update_ix[17]
        #    
        #    self.test_learning_batch = upper_bound - lower_bound
        #    # THIS NEEDS TO BE FIXED...
        #    s_temp = self.training_data[lower_bound:upper_bound,:]
        #    if self.normalize_EMG:
        #        s_normed = s_temp/np.amax(s_temp)
        #    else:
        #        s_normed = s_temp
        #    if self.PCA_comps!=self.pca_channel_default:  
        #        pca = PCA(n_components=self.PCA_comps)
        #        s_normed = pca.fit_transform(s_normed)
        #    self.s_test = np.transpose(s_normed)
        #    self.F_test = self.s_test[:,:-1] # note: truncate F for estimate_decoder
        #    self.p_test_reference = np.transpose(self.labels[lower_bound:upper_bound,:])
        #elif self.use_up16_for_test==True and final_eval==False:
        #    # Can reuse the test values already set earlier in the init func
        #    pass
        #else:
        #    raise ValueError("use_up16_for_test must be True, it is the only testing supported currently")
            
        if type(self.F_test) is type([]):
            running_test_loss = 0
            running_num_samples = 0
            for i in range(len(self.F_test)):
                # TODO: Ensure that this is supposed to be D@s and not D@F...
                v_actual = model@self.s_test[i]
                p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
                V_test = (self.p_test_reference[i] - p_actual)*self.dt
                batch_loss = cost_l2(self.F_test[i], model, V_test, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
                batch_samples = self.F_test[i].shape[0]
                running_test_loss += batch_loss * batch_samples  # Accumulate weighted loss by number of samples in batch
                running_num_samples += batch_samples
            normalized_test_loss = running_test_loss / running_num_samples  # Normalize by total number of samples
            test_log.append(normalized_test_loss)
        else:
            # TODO: Ensure that this is supposed to be D@s and not D@F...
            v_actual = model@self.s_test
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            V_test = (self.p_test_reference - p_actual)*self.dt
            test_loss = cost_l2(self.F_test, model, V_test, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
            total_samples = self.F_test.shape[1]  # Shape is (64, 1201)
            normalized_test_loss = test_loss / total_samples  # Normalize by the number of samples
            test_log.append(normalized_test_loss)
        
        # This was returning self.V_test for some reason
        ## I have None as a placeholder for now, maybe I'll return testing samples? But it already has access to that...
        return normalized_test_loss, None