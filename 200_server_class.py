class Server(ModelBase):
    def __init__(self, ID, D0, method, all_clients, smoothbatch=1, C=0.1, normalize_dec=True, current_round=0, PCA_comps=7, verbose=False, experimental_plotting=False, num_steps=10):
        super().__init__(ID, D0, method, smoothbatch=smoothbatch, current_round=current_round, PCA_comps=PCA_comps, verbose=verbose, num_participants=14, log_init=0)
        self.type = 'Server'
        self.num_avail_clients = 0
        self.available_clients_lst = [0]*len(all_clients)
        self.num_chosen_clients = 0
        self.chosen_clients_lst = [0]*len(all_clients)
        self.all_clients = all_clients
        self.C = C  # Fraction of clients to use each round
        self.experimental_plotting = False
        self.experimental_inclusion_round = [0]
        self.init_lst = [self.log_init]*self.num_participants
        self.normalize_dec = normalize_dec
        self.set_available_clients_list(init=True)
        if self.method=='APFL':
            self.set_available_clients_list()
            self.choose_clients()
            self.K = len(self.chosen_clients_lst)
            # NOTE: TAU IS USED OVER CLIENT'S NUM_STEPS FOR APFL
            self.tau = num_steps

                
    # 0: Main Loop
    def execute_FL_loop(self):
        # Update global round number
        self.current_round += 1
        
        if self.method=='FedAvg':
            # Choose fraction C of available clients
            self.set_available_clients_list()
            self.choose_clients()
            # Send those clients the current global model
            if self.experimental_plotting:
                for my_client in self.available_clients_lst:
                    if my_client in self.chosen_clients_lst:
                        my_client.global_w = self.w
                    else:
                        my_client.local_error_log.append(my_client.local_error_log[-1])
                        my_client.global_error_log.append(my_client.global_error_log[-1])    
                        my_client.personalized_error_log.append(my_client.personalized_error_log[-1])    
            else:
                for my_client in self.chosen_clients_lst:
                    my_client.global_w = self.w
            # Let those clients train (this autoselects the chosen_client_lst to use)
            self.train_client_and_log(client_set=self.chosen_clients_lst)
            # AGGREGATION
            self.w_prev = copy.copy(self.w)
            self.agg_local_weights()  # This func sets self.w, eg the new decoder
            # GLOBAL SmoothBatch
            #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
            self.w = self.smoothbatch*self.w + ((1 - self.smoothbatch)*self.w_prev)
            #^ Note when self.smoothbatch=1 (default), we just keep the new self.w (no mixing)
        elif self.method=='NoFL':
            self.train_client_and_log(client_set=self.all_clients)
        elif self.method=='APFL':
            t = self.current_round
            
            running_dec_aggr = 0
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
                #    my_client.personalized_error_log.append(self.personalized_error_log[-1])
                
                # Aggregate global dec every tau iters
                running_global_dec = 0
                for my_client in self.chosen_clients_lst:
                    running_global_dec += my_client.global_w
                self.w = copy.copy(running_global_dec)
                self.set_available_clients_list()
                self.choose_clients()
                # Presumably len(self.chosen_clients_lst) will always be the same (same C)... 
                # otherwise would need to re-set K as so:
                #self.K = len(self.chosen_clients_lst)

                for my_client in self.chosen_clients_lst:
                    my_client.global_w = copy.copy(self.w)  
                    #^ Is copy necessary? Depends if updated or overwritten...
            self.w = (1/self.K)*running_dec_aggr
        else:
            raise('Method not currently supported, please reset method to FedAvg')
        # Save the new decoder to the log
        self.dec_log.append(self.w)
        # Reset all clients so no one is chosen for the next round
        for my_client in self.all_clients:  # Would it be better to use just the chosen_clients or do all_clients?
            if type(my_client)==int:
                raise("All my clients are all integers...")
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
                    #if self.method != 'NoFL':
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
                raise(f"ERROR: Chose {self.num_chosen_clients} clients for some reason, must choose more than 1")
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
                if self.method == 'APFL':
                    pers_init_carry_val = self.personalized_error_log[-1][my_client.ID][2]#[0]
            if my_client in client_set:
                my_client.execute_training_loop()
                current_local_lst.append((my_client.ID, self.current_round, 
                                          my_client.eval_model(which='local')))
                if self.method != 'NoFL':
                    current_global_lst.append((my_client.ID, self.current_round, 
                                               my_client.eval_model(which='global')))
                if self.method == 'APFL':
                    current_pers_lst.append((my_client.ID, self.current_round, 
                                               my_client.eval_model(which='pers')))
            else:
                current_local_lst.append((my_client.ID, self.current_round, 
                                          local_init_carry_val))
                if self.method != 'NoFL':
                    current_global_lst.append((my_client.ID, self.current_round,
                                               global_init_carry_val))
                if self.method == 'APFL':
                    current_pers_lst.append((my_client.ID, self.current_round,
                                               pers_init_carry_val))
        # Append (ID, COST) to SERVER'S error log.  
        #  Note that round is implicit, it is just the index of the error log
        if self.local_error_log==self.init_lst:
            # Overwrite the [(0,0)] hold
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
        if self.method == 'APFL':
            if self.personalized_error_log==self.init_lst:
                self.personalized_error_log = []
                self.personalized_error_log.append(current_pers_lst)
            else:
                self.personalized_error_log.append(current_pers_lst)
    
    # 3
    def agg_local_weights(self):
        # From McMahan 2017 (vanilla FL)
        summed_num_datapoints = 0
        for my_client in self.chosen_clients_lst:
            summed_num_datapoints += my_client.learning_batch
        # Aggregate local model weights, weighted by normalized local learning rate
        aggr_w = 0
        for my_client in self.chosen_clients_lst:
            if self.normalize_dec:
                normalization_term = np.amax(my_client.w)
            else:
                normalization_term = 1
            aggr_w += (my_client.learning_batch/summed_num_datapoints) * my_client.w / normalization_term
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