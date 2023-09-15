class Client(ModelBase, TrainingMethods):
    def __init__(self, ID, w, method, local_data, data_stream, smoothbatch=1, current_round=0, PCA_comps=7, availability=1, global_method='FedAvg', normalize_dec=False, normalize_EMG=True, starting_update=0, track_cost_components=True, track_lr_comps=True, use_real_hess=True, gradient_clipping=False, log_decs=True, clipping_threshold=100, tol=1e-10, adaptive=True, eta=1, track_gradient=True, wprev_global=False, num_steps=1, use_zvel=False, APFL_input_eta=False, safe_lr_factor=False, set_alphaF_zero=False, mix_in_each_steps=False, mix_mixed_SB=False, delay_scaling=5, random_delays=False, download_delay=1, upload_delay=1, copy_type='deep', validate_memory_IDs=True, local_round_threshold=25, condition_number=1, verbose=False):
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
        test_split_product_index = local_data['training'].shape[0]*test_split_frac
        # Convert this value to the cloest update_ix value
        train_test_update_number_split = min(self.update_ix, key=lambda x:abs(x-test_split_product_index))
        self.test_split_idx = update_ix[train_test_update_number_split]
        print(f"use_up16_for_test: {self.use_up16_for_test}")
        print(f"test_split_product_index = local_data['training'].shape[0]*test_split_frac = {test_split_product_index}")
        print(f"train_test_update_number_split (closest update): {train_test_update_number_split}")
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
            lower_bound = (update_ix[15])//2  #Use only the second half of each update
            upper_bound = update_ix[16]
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