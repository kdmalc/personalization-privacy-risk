class Client(ModelBase, TrainingMethods):
    def __init__(self, ID, w, method, local_data, data_stream, smoothbatch=1, current_round=0, PCA_comps=7, availability=1, global_method='FedAvg', normalize_dec=False, track_cost_components=True, use_real_hess=False, gradient_clipping=False, clipping_threshold=100, tol=1e-10, adaptive=True, eta=1, track_gradient=True, num_steps=1, input_eta=False, safe_lr=False, delay_scaling=5, normalize_EMG=True, random_delays=False, download_delay=1, upload_delay=1, local_round_threshold=50, condition_number=0, verbose=False):
        super().__init__(ID, w, method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, verbose=verbose, num_participants=14, log_init=0)
        '''
        Note self.smoothbatch gets overwritten according to the condition number!  
        If you want NO smoothbatch then set it to 'off'
        '''
        # NOT INPUT
        self.type = 'Client'
        self.chosen_status = 0
        self.current_global_round = 0
        self.update_transition_log = []
        self.normalize_EMG = normalize_EMG
        # Sentinel Values
        self.F = None
        self.V = None
        self.D = None
        self.H = np.zeros((2,2))
        self.learning_batch = None
        self.dt = 1.0/60.0
        self.eta = eta
        self.training_data = local_data['training']
        self.labels = local_data['labels']
        # Round minimization output to the nearest int or keep as a float?  Don't need arbitrary precision
        self.round2int = False
        self.normalize_dec = normalize_dec
        # FL CLASS STUFF
        # Availability for training
        self.availability = availability
        # Toggle streaming aspect of data collection: {Ignore updates and use all the data; 
        #  Stream each update, moving to the next update after local_round_threshold iters have been run; 
        #  After 1 iteration, move to the next update}
        self.data_stream = data_stream  # {'full_data', 'streaming', 'advance_each_iter'} 
        # Number of gradient steps to take when training (eg amount of local computation)
        self.num_steps = num_steps
        # GLOBAL STUFF
        self.global_w = None
        self.global_method = global_method
        # UPDATE STUFF
        if self.global_method=='NoFL':
            starting_update = 0
        else:
            starting_update = self.cphs_starting_update
        self.current_update = starting_update
        self.local_round_threshold = local_round_threshold
        self.current_threshold = local_round_threshold
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
        self.alphaE = 1e-6
        # This should probably be a dictionary at some point
        if condition_number==0:
            self.smoothbatch = 0.25
            self.alphaF = 1e-7
            self.alphaD = 1e-3
        else:
            print("That condition number is not yet supported")
        if smoothbatch=='off':
            self.smoothbatch = 1  # AKA Use only the new dec, no mixing
        # PLOTTING
        # ^Eg the local round is implicit, it can't have a skip in round
        # Overwrite the logs since global and local track in slightly different ways
        self.local_error_log = [0]
        self.global_error_log = [0]
        self.personalized_error_log = [0]
        self.performance_log = [0]
        self.Dnorm_log = [0]
        self.Fnorm_log = [0]
        self.gradient_log = [0]
        self.track_cost_components = track_cost_components
        self.track_gradient = track_gradient
        # APFL Stuff
        self.tol = tol
        self.input_eta = input_eta
        self.running_pers_term = 0
        self.running_global_term = 0
        self.global_w = copy.copy(self.w)
        self.local_w = copy.copy(self.w)
        self.mixed_w = copy.copy(self.w)
        self.adaptive = adaptive
        #They observed best results with 0.25, but initialized adap_alpha to 0.01 for the adaptive case
        if self.adaptive:
            self.adap_alpha = [0.01]  
        else:
            self.adap_alpha = [0.25] 
        self.final_personalized_w = None
        self.final_global_w = None
        self.tau = self.num_steps
        self.p = [0]
        self.safe_lr = safe_lr
        self.gradient_clipping = gradient_clipping
        self.clipping_threshold = clipping_threshold
        self.Vlocal = None
        self.Vglobal = None
        #if self.global_method=='APFL':
        #    self.w = None  # This is a massive issue....
        self.use_real_hess = use_real_hess
        #
        self.prev_update = None
        self.prev_eigvals = None
                                                               
    # 0: Main Loop
    def execute_training_loop(self):
        self.simulate_data_stream()
        self.train_model()
        
        # Append (ROUND, COST) to the CLIENT error log
        local_loss = self.eval_model(which='local')
        self.local_error_log.append(local_loss)  # ((self.current_round, local_loss))
        # Yes these should both be ifs, they may both need to run
        if self.global_method!="NoFL":
            global_loss = self.eval_model(which='global')
            self.global_error_log.append(global_loss)  # ((self.current_round, global_loss))
        if self.global_method=="APFL":
            pers_loss = self.eval_model(which='pers')
            self.personalized_error_log.append(pers_loss)  # ((self.current_round, pers_loss))
        D = self.mixed_w if self.global_method=='APFL' else self.w
        if self.track_cost_components:
            self.performance_log.append(self.alphaE*(np.linalg.norm((D@self.F + self.H@self.V[:,:-1] - self.V[:,1:]))**2))
            self.Dnorm_log.append(self.alphaD*(np.linalg.norm(D)**2))
            self.Fnorm_log.append(self.alphaF*(np.linalg.norm(self.F)**2))
        if self.track_gradient:
            # The gradient is a vector... So let's just save the L2 norm?
            self.gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, D, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)))

        
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
            # Probably ought to track that we maxed out
            #lower_bound = update_ix[-2]  
            # ^Used to be 0 (e.g. full dataset instead of last update), saw bad behaviour...
            # We are stopping an update early, so use -3/-2 and not -2/-1 (the last update)
            lower_bound = (update_ix[-3] + update_ix[-2])//2  #Use only the second half of each update
            upper_bound = update_ix[-2]
            self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='full_data':
            #print("FULL")
            lower_bound = update_ix[0]  # Starts at 0 and not update 10, for now
            upper_bound = update_ix[-1]
            self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='streaming':
            #print("STREAMING")
            if self.current_round >= self.current_threshold:
                self.current_threshold += self.local_round_threshold
                #self.current_threshold += self.current_threshold  # This is the "wrong" doubling version
                
                ####################################################
                self.prev_update = self.current_update
                self.current_update = self.prev_update + 1
                
                self.update_transition_log.append(self.current_global_round)
                if self.verbose==True and self.ID==1:
                    print(f"Client {self.ID}: New update after lrt passed: (new update, current global round, current local round): {self.current_update, self.current_global_round, self.current_round}")
                    print()
                    
                # Using only the second half of each update for co-adaptivity reasons
                lower_bound = (update_ix[self.current_update] + update_ix[self.current_update+1])//2  
                upper_bound = update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
            elif self.current_round>2:  # Allow the init condition to still run
                # The update number didn't change so we don't need to overwrite everything with the same data
                need_to_advance = False
            else:
                # Using only the second half of each update for co-adaptivity reasons
                lower_bound = (update_ix[self.current_update] + update_ix[self.current_update+1])//2  
                upper_bound = update_ix[self.current_update+1]
                self.learning_batch = upper_bound - lower_bound
        elif streaming_method=='advance_each_iter':
            #print("ADVANCE")
            #lower_bound = update_ix[self.current_update]
            lower_bound = (update_ix[self.current_update] + update_ix[self.current_update+1])//2  
            #^Use only the second half of each update
            upper_bound = update_ix[self.current_update+1]
            self.learning_batch = upper_bound - lower_bound
            
            ####################################################
            self.prev_update = self.current_update
            self.current_update = self.prev_update + 1
        else:
            raise ValueError('This data streaming functionality is not supported')
            
        if need_to_advance:
            s_temp = self.training_data[lower_bound:upper_bound,:]
            # First, normalize the entire s matrix
            # Hope is that this will prevent FF.T from being massive in APFL
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
            v_actual = self.w@s
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            p_reference = np.transpose(self.labels[lower_bound:upper_bound,:])
            # Now set the values used in the cost function
            self.F = s[:,:-1] # note: truncate F for estimate_decoder
            #self.V = (np.transpose(self.labels[lower_bound:upper_bound,:]) - np.cumsum(self.w@s, axis=1)*self.dt)*self.dt
            self.V = (p_reference - p_actual)*self.dt
            if self.global_method=='APFL':
                self.Vglobal = (np.transpose(self.labels[lower_bound:upper_bound,:]) - np.cumsum(self.global_w@s, axis=1)*self.dt)*self.dt
                self.Vlocal = (np.transpose(self.labels[lower_bound:upper_bound,:]) - np.cumsum(self.local_w@s, axis=1)*self.dt)*self.dt
                #^ Should this be local or mixed? I think local... even though it is evaluated at the mixed dec... not sure
            self.D = copy.copy(self.w)
    
    def train_model(self):
        D_0 = self.w_prev
        # Set the w_prev equal to the current w:
        self.w_prev = self.w
        if self.global_method=="FedAvg":
            # Overwrite local model with the new global model
            self.w = self.global_w
        if self.global_method=="FedAvg" or self.global_method=="NoFL":
            for i in range(self.num_steps):
                ########################################
                # Should I normalize the dec here?  
                # I think this will prevent it from blowing up if I norm it every time
                if self.normalize_dec:
                    self.w /= np.amax(self.w)
                ########################################
                if self.method=='EtaGradStep':
                    self.w = self.train_eta_gradstep(self.w, self.eta, self.F, self.D, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, PCA_comps=self.PCA_comps)
                elif self.method=='EtaScipyMinStep':
                    self.w = self.train_eta_scipyminstep(self.w, self.eta, self.F, self.D, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, D_0, self.verbose, PCA_comps=self.PCA_comps)
                elif self.method=='FullScipyMinStep':
                    self.w = self.train_eta_scipyminstep(self.w, self.eta, self.F, self.D, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, D_0, self.verbose, PCA_comps=self.PCA_comps, full=True)
                else:
                    raise ValueError("Unrecognized method")
            ########################################
            # Or should I normalize the dec here?  I'll also turn this on since idc about computational speed rn
            if self.normalize_dec:
                self.w /= np.amax(self.w)
            ########################################
            # Do SmoothBatch
            # Maybe move this to only happen after each update? Does it really need to happen every iter?
            # I'd have to add weird flags just for this in various places... put on hold for now
            #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
            self.w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.w_prev)
        elif self.global_method=='APFL': 
            t = self.current_global_round  # Should this be global or local? Global based on how they wrote it...
            # Is it just F or did Cesar forget/ignore the other smaller terms?
            # F.T@F is not symmetric, so use eig not eigh
            # eig returns (UNORDERED) eigvals, eigvecs
            if self.use_real_hess:
                if self.prev_update == self.current_update:
                    # Note that this changes if you want to do SGD instead of GD
                    eigvals = self.prev_eigvals
                else:
                    print(f"Client{self.ID}: Recalculating the Hessian for new update {self.current_update}!")
                    eigvals, _ = np.linalg.eig(hessian_cost_l2(self.F, self.alphaD))
                    self.prev_eigvals = eigvals
            else:
                eigvals, _ = np.linalg.eig(self.F.T@self.F)
            mu = np.amin(eigvals)  # Mu is the minimum eigvalue
            if mu.imag < self.tol and mu.real < self.tol:
                #mu = 0
                # Implies it is not mu-strongly convex
                #print(f"mu ({mu}) is effectively 0... resetting to self.alphaD: {self.alphaD}")
                #raise ValueError("mu is 0")
                
                # Fudge factor... based off my closed form solution...
                # Really ought to create a log file/system
                mu = self.alphaD
            elif mu.imag < self.tol:
                mu = mu.real
            elif mu.real < self.tol:
                print("Setting to imaginary only")
                mu = mu.imag
            L = np.amax(eigvals)  # L is the maximum eigvalue
            if L.imag < self.tol and L.real < self.tol:
                raise ValueError("L is 0, thus implying func is not L-smooth")
            elif mu.imag < self.tol:
                L = L.real
            elif L.real < self.tol:
                print("Setting to imaginary only")
                L = L.imag
            if self.verbose:
                print(f"ID: {self.ID}, L: {L}, mu: {mu}")
            kappa = L/mu
            a = np.max([128*kappa, self.tau])
            eta_t = 16 / (mu*(t+a))
            if self.input_eta:
                if self.safe_lr!=False:
                    raise ValueError("Cannot input eta AND use safe learning rate (they overwrite each other)")
                eta_t = self.eta
            elif self.safe_lr!=False:
                eta_t = 1/(self.safe_lr*L)
            elif eta_t >= 1/(2*L):
                # Note that we only check when automatically setting
                # ie if you manually input it will do whatever you tell it to do
                raise ValueError("Learning rate is too large according to constaints on GD")
            if self.verbose:
                print(f"ID: {self.ID}, eta_t: {eta_t}")
                print()
            self.p.append((t+a)**2)
            
            if self.adaptive:
                self.adap_alpha.append(self.adap_alpha[-1] - eta_t*np.inner(np.reshape((self.local_w-self.global_w), (self.PCA_comps*2)), np.reshape(gradient_cost_l2(self.F, self.mixed_w, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps), (2*self.PCA_comps,))))
                # This is theoretically the same but I'm not sure what grad_alpha means
                #self.sus_adap_alpha.append() ... didn't write yet

            # GRADIENT DESCENT BASED MODEL UPDATE
            # NOTE: eta_t IS DIFFERENT FROM CLIENT'S ETA (WHICH IS NOT USED)
            # I think the grads really ought to be reshaping this automatically, not sure why it's not
            
            global_gradient = np.reshape(gradient_cost_l2(self.F, self.global_w, self.H, self.Vglobal, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps), (2, self.PCA_comps))
            local_gradient = np.reshape(gradient_cost_l2(self.F, self.mixed_w, self.H, self.Vlocal, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps), (2, self.PCA_comps))
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
                self.local_w /= np.amax(self.local_w)
                self.mixed_w /= np.amax(self.mixed_w)
            ########################################
            
            # PSEUDOCODE: my_client.global_w -= my_client.eta * grad(f_i(my_client.global_w; my_client.smallChi))
            self.global_w -= eta_t * global_gradient
            # PSEUDOCODE: my_client.local_w -= my_client.eta * grad_v(f_i(my_client.v_bar; my_client.smallChi))
            self.local_w -= eta_t * local_gradient
            self.mixed_w = self.adap_alpha[-1]*self.local_w - (1 - self.adap_alpha[-1])*self.global_w
            ########################################
            # Or should I normalize the dec here?  I'll also turn this on since idc about computational speed rn
            if self.normalize_dec:
                self.global_w /= np.amax(self.global_w)
                self.local_w /= np.amax(self.local_w)
                self.mixed_w /= np.amax(self.mixed_w)
            ########################################
        # Save the new decoder to the log
        self.dec_log.append(self.mixed_w)
        
    def eval_model(self, which):
        if which=='local':
            my_dec = self.w
            #my_error_log = self.local_error_log
        elif which=='global':
            my_dec = self.global_w
            #my_error_log = self.global_error_log
        elif which=='pers' and self.global_method=='APFL':
            my_dec = self.mixed_w
        else:
            print("Please set <which> to either local or global")
        # Just did this so we wouldn't have the 14 decimals points it always tries to give
        if self.round2int:
            temp = np.ceil(cost_l2(self.F, my_dec, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD))
            # Setting to int is just to catch overflow errors
            # For RT considerations, ints are also generally ints cheaper than floats...
            out = int(temp)
        else:
            temp = cost_l2(self.F, my_dec, self.H, self.V, self.learning_batch, self.alphaF, self.alphaD, Ne=self.PCA_comps)
            out = round(temp, 3)
        return out
        
    def test_inference(self, test_dec=True):
        ''' No training / optimization, this just tests the fed in dec '''
        
        if test_dec:
            test_dec = self.w
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