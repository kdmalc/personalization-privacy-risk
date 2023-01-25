import numpy as np


# set up gradient of cost:
# d(c_L2(D))/d(D) = 2*(DF + HV - V+)*F.T + 2*alphaD*D
def gradient_cost_l2(F, D, H, V, alphaF=1e-2, alphaD=1e-2):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder # TODO: we now have a timeseries component - consult Sam
    H: 2 x 2 state transition matrix
    ''' 
    Nd = 2
    Ne = 64
    Nt = learning_batch

    # TODO: add depth (time) to D
    D = np.reshape(D,(Nd, Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]

    return ((2 * (D@F + H@Vminus - Vplus) @ F.T / (Nd*Nt) 
        + 2 * alphaD * D / (Nd*Ne)).flatten())


# set up gradient of cost:
# d(c_L2(D))/d(D) = 2*(DF + HV - V+)*F.T + 2*alphaD*D
def gradient_cost_l2_discrete(F, D, H, V, alphaF=1e-2, alphaD=1e-2):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder # TODO: we now have a timeseries component - consult Sam
    H: 2 x 2 state transition matrix
    ''' 
    Nd = 2
    Ne = 64
    Nt = learning_batch

    # TODO: add depth (time) to D
    D = np.reshape(D,(Nd, Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]
    v_unbounded = D@F
    theta = np.arctan2(v_unbounded[1],v_unbounded[0])
    v_actual = np.asarray([10*np.cos(theta),10*np.sin(theta)])
    return ((2 * (v_actual + H@Vminus - Vplus) @ F.T / (Nd*Nt) 
        + 2 * alphaD * D / (Nd*Ne)).flatten())

    
# set up the cost function: 
# c_L2 = (||DF + HV - V+||_2)^2 + alphaD*(||D||_2)^2 + alphaF*(||F||_2)^2
def cost_l2_discrete(F, D, H, V, alphaF=1e-2, alphaD=1e-2):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder
    H: 2 x 2 state transition matrix
    ''' 
    Nd = 2
    Ne = 64 # default = 64
    Nt = learning_batch
    # TODO: add depth (time) to D
    D = np.reshape(D,(Nd,Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]
    v_unbounded = D@F
    theta = np.arctan2(v_unbounded[1],v_unbounded[0])
    v_actual = np.asarray([10*np.cos(theta),10*np.sin(theta)])
    e = ( np.sum( (v_actual + H@Vminus - Vplus)**2 ) / (Nd*Nt) 
            + alphaD * np.sum( D**2 ) / (Nd*Ne)
            + alphaF * np.sum( F**2 ) / (Ne*Nt) )
    return e


# set up the cost function: 
# c_L2 = (||DF + HV - V+||_2)^2 + alphaD*(||D||_2)^2 + alphaF*(||F||_2)^2
def cost_l2(F, D, H, V, alphaF=1e-2, alphaD=1e-2):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder
    H: 2 x 2 state transition matrix
    ''' 
    Nd = 2
    Ne = 64 # default = 64
    Nt = learning_batch
    # TODO: add depth (time) to D
    D = np.reshape(D,(Nd,Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]

    e = ( np.sum( (D @ F + H@Vminus - Vplus)**2 ) / (Nd*Nt) 
            + alphaD * np.sum( D**2 ) / (Nd*Ne)
            + alphaF * np.sum( F**2 ) / (Ne*Nt) )
    return e


def estimate_decoder(F, H, V):
    return (V[:,1:]-H@V[:,:-1])@np.linalg.pinv(F)


# Added 2 new parameters
def simulation(D,learning_batch,alpha,alphaF=1e-2,alphaD=1e-2,display_info=False,num_iters=False):
    p_classify = []
    accuracy_temp = []
    num_updates = int(np.floor((filtered_signals.shape[0]-1)/learning_batch)) # how many times can we update decoder based on learning batch    

    # RANDOMIZE DATASET
    randomized_integers = np.random.permutation(range(0,cued_target_position.shape[0]))
    filtered_signals_randomized = filtered_signals[randomized_integers]
    cued_target_position_randomized = cued_target_position[randomized_integers]
    # batches the trials into each of the update batch
    for ix in range(num_updates):
        s = np.hstack([x for x in filtered_signals_randomized[int(ix*learning_batch+1):int((ix+1)*learning_batch+1),:,:]])# stack s (64 x (60 timepoints x learning batch size))
        p_intended = np.hstack([np.tile(x[:,np.newaxis],60) for x in cued_target_position_randomized[int(ix*learning_batch+1):int((ix+1)*learning_batch+1),:]]) # stack p_intended (2 x 60 timepoints x learning batch size)
        v_intended,p_constrained = output_new_decoder(s,D[-1],p_intended)

        # CLASSIFY CURRENT DECODER ACCURACY
        v_actual = D[-1]@s
        for trial in range(learning_batch):
            v_trial = v_actual[:,int(trial*60):int((trial+1)*60)] # velocities for each trials (2,60)
            p_final = np.sum(v_trial,axis=1)[:,np.newaxis] # final position after integration (2,)
            p_classify.append(classify(p_final))
        
        # UPDATE DECODER
        u = copy.deepcopy(s) # u is the person's signal s (64 CHANNELS X TIMEPOINTS)
        q = copy.deepcopy(v_intended) # use cued positions as velocity vectors for updating decoder should be 2 x num_trials

        # emg_windows against intended_targets (trial specific cued target)
        F = copy.deepcopy(u[:,:-1]) # note: truncate F for estimate_decoder
        V = copy.deepcopy(q)

        # initial decoder estimate for gradient descent
        D0 = np.random.rand(2,64)

        # set alphas
        H = np.zeros((2,2))
        # use scipy minimize for gradient descent and provide pre-computed analytical gradient for speed
        if num_iters is False:
            out = minimize(lambda D: cost_l2(F,D,H,V), D0, method='BFGS', jac = lambda D: gradient_cost_l2(F,D,H,V), options={'disp': display_info})
        else:
            out = minimize(lambda D: cost_l2(F,D,H,V), D0, method='BFGS', jac = lambda D: gradient_cost_l2(F,D,H,V), options={'disp': display_info, 'maxiter':num_iters})
        
        # reshape to decoder parameters
        W_hat = np.reshape(out.x,(2, 64))

        # DO SMOOTHBATCH
        W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
        D.append(W_new)

        # COMPUTE CLASSIFICATION ACURACY 
        p_target = (cued_target_position[randomized_integers])[int(ix*learning_batch+1):int((ix+1)*learning_batch+1),:] # obtain target
        accuracy_temp.append(classification_accuracy(p_target,p_classify[-learning_batch:]))

    p_classify = np.asarray(p_classify)
    return accuracy_temp,D,p_constrained


# constrain p - do we want to include radius distance when we update the new decoder?
def constrain_p_actual(p):
    '''
    input: decoded velocity (2,)
    output: constrained decoded velocity (2,)
    '''
    # if np.linalg.norm(p) >= 10:
    theta = np.arctan2(p[1],p[0])
    return [10*np.cos(theta),10*np.sin(theta)]
    # else:
    #     return p


# integrate velocity 
def compute_position(v):
    '''
    input v: velocity (2,T)
    output: position (2,)
    '''
    return np.sum(v,axis=1)


# given decoder, what is the new position?
def output_new_decoder(s,D,p_intended):
    '''
    s: (64 x (60 timepoints x learning batch size))
    D: (2 x 64) previous D computed or random
    p_intended: (2 x 60 timepoints x learning batch size)
    '''
    # take first trial, random decoder, and do target classification
    v = D@s # actual decoded velocity (2,60)

    # integrate decoded velocities into positions
    p = []
    for ix in range(v.shape[1]):
        p.append(np.sum(v[:,:ix],axis=1))
    p = np.asarray(p).T # actual decoded position (2,60)

    # want error between intended and actual velocity but need to constrain actual velocity to target radius
    p_constrained = np.asarray([constrain_p_actual(p_) for p_ in p.T]).T # constrained, (2,60)
    # compute error between intended and actual position and take derivative to get intended velocity
    v_intended = p_intended - p_constrained # (2,60)
    return v_intended,p_constrained


# given decoder, what is the new position?
def output_new_decoder_constant_intention(s,D,p_intended):
    '''
    s: (64 x (60 timepoints x learning batch size))
    D: (2 x 64) previous D computed or random
    p_intended: (2 x 60 timepoints x learning batch size)
    '''
    # take first trial, random decoder, and do target classification
    v = D@s # actual decoded velocity (2,60)

    # integrate decoded velocities into positions
    p = []
    for ix in range(v.shape[1]):
        p.append(np.sum(v[:,:ix],axis=1))
    p = np.asarray(p).T # actual decoded position (2,60)

    # want error between intended and actual velocity but need to constrain actual velocity to target radius
    p_constrained = np.asarray([constrain_p_actual(p_) for p_ in p.T]).T # constrained, (2,60)
    # compute error between intended and actual position and take derivative to get intended velocity
    v_intended = p_intended # (2,60) # divide by 60 was removed
    return v_intended,p_constrained


def classify(decoded_cursor_velocity, print_indx = False):
    
    # pick the smallest distance diff
    dist_diffs = np.linalg.norm((decoded_cursor_velocity.T - target_positions),
                                axis = 1) # this should be an array
    
    min_dist_diffs = np.argmin(dist_diffs)

    classified_target = target_positions[min_dist_diffs,:]#[:, None]

    # target_postions is a global variabe
    return classified_target


def classification_accuracy(p_target,p_classify): 
    '''
    inputs: 
        p_target: target positions (trials x 2)
        p_classify: classified positions (trials x 2)
    output:
        success_rate: classification accuracy (number)
    '''
    target_pos_diff = p_classify - p_target
    target_pos_diff_norm = np.linalg.norm(target_pos_diff, axis = 1)
    target_pos_count = sum(target_pos_diff_norm < 0.01)
    success_rate = target_pos_count / len(target_pos_diff_norm)
    return success_rate