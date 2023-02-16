import numpy as np


# set up gradient of cost:
# d(c_L2(D))/d(D) = 2*(DF + HV - V+)*F.T + 2*alphaD*D
def gradient_cost_l2(F, D, H, V, learning_batch, alphaF, alphaD, alphaE=1e-6, Nd=2, Ne=64):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder # TODO: we now have a timeseries component - consult Sam
    H: 2 x 2 state transition matrix
    
    alphaE is 1e-6 for all conditions
    ''' 
    
    Nt = learning_batch

    D = np.reshape(D,(Nd, Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]

    return ((2 * (D@F + H@Vminus - Vplus)@F.T*(alphaE) #/ (Nd*Nt) # They multiply F.T by lambdaE
        + 2*alphaD*D ).flatten())  #/ (Nd*Ne)


# set up the cost function: 
# c_L2 = (||DF + HV - V+||_2)^2 + alphaD*(||D||_2)^2 + alphaF*(||F||_2)^2
def cost_l2(F, D, H, V, learning_batch, alphaF, alphaD, alphaE=1e-6, Nd=2, Ne=64):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder
    H: 2 x 2 state transition matrix
    
    alphaE is 1e-6 for all conditions
    ''' 

    Nt = learning_batch
    D = np.reshape(D,(Nd,Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]

    #e = ( np.sum((D@F + H@Vminus - Vplus)**2)*(alphaE) #/ (Nd*Nt) 
    #        + alphaD*np.sum(D**2) #/ (Nd*Ne)
    #        + alphaF*np.sum(F**2) ) #/ (Ne*Nt) )
    
    term1 = np.sum((D@F + H@Vminus - Vplus)**2)*(alphaE)
    #term2 = alphaD*np.sum(D**2) #/ (Nd*Ne)
    #term3 = alphaF*np.sum(F**2) #/ (Ne*Nt) )
    term2 = alphaD*np.linalg.norm(D)
    term3 = alphaF*np.linalg.norm(F)
    
    return (term1 + term2 + term3)


def estimate_decoder(F, H, V):
    return (V[:,1:]-H@V[:,:-1])@np.linalg.pinv(F)


# Added 2 new parameters; use the up-to-date code in NB 200
# Otherwise this is the original sims code.  It uses the discrete trial funcs still I believe
'''
def simulation(D, learning_batch, alpha, alphaF=1e-2, alphaD=1e-2, display_info=False, num_iters=False):
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
'''