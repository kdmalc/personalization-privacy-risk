# This is just a referene, verbatim code pulled from my fl_sim_classes

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