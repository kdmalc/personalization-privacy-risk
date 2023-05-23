import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from utils.custom_loss_func import CPHSLoss

def cost_l2_torch(F, D, V, learning_batch, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False):
    # c_L2 = (lambdaE||DF + V+||_2)^2 + lambdaD*(||D||_2)^2 + lambdaF*(||F||_2)^2
    
    # Don't use return_cost_func_comps since I don't think loss.item() will return a tuple, it only returns scalaras AFAIK
    
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder
    H: 2 x 2 state transition matrix
    alphaE is 1e-6 for all conditions
    ''' 
    
    # Hmm should I detach and use numpy or just stick with tensor ops?
    # I don't want gradient to be tracked here but idk if it matters...

    Nt = learning_batch
    D = D.view(Nd, Ne)  #np.reshape(D,(Nd,Ne))
    Vplus = V[:,1:]
    # Performance
    term1 = lambdaE*(torch.linalg.matrix_norm((torch.matmul(D, F) - Vplus))**2)
    # D Norm
    term2 = lambdaD*(torch.linalg.matrix_norm((D)**2))
    # F Norm
    term3 = lambdaF*(torch.linalg.matrix_norm((F)**2))
    return (term1 + term2 + term3)

def full_train_linregr_updates(model, full_trial_input_data, full_trial_labels, learning_rate, lambdasFDE=[0, 1e-3, 1e-6], use_CPHSLoss=False, normalize_emg=False, PCA_comps=64, num_iters_per_update=30, normalize_V=False, starting_update=10, use_full_input_data=False, stream_data_updates=True, dt=1/60, loss_log=None, verbose=False, verbose_norms=False, return_cost_func_comps=False, update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]):
    
    ##################
    ETerm_log = []
    DTerm_log = []
    ##################
    
    if loss_log is None:
        loss_log = list()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
    if use_full_input_data:
        num_updates = 1  # e.g. only complete 1 loop, using the full data
    else:
        num_updates = len(update_ix[starting_update:])-1-1
        #-1 to account for the update+1 we do for upper bound
        #-1 again to account for us wanting to skip the last update since it is truncated
    if verbose:
        print(f"Num_updates: {num_updates}")
        
    for update in range(num_updates):  
        update += starting_update
        if verbose:
            print(f"Current Update: {update}")
        
        if use_full_input_data:
            lower_bound = 0
            upper_bound = -1
        #elif use_full_data:
        #    lower_bound = update_ix[starting_update]
        #    upper_bound = update_ix[-1]  # Could set this to be -2 and allow skipping the truncated segment...
        elif stream_data_updates:
            lower_bound = update_ix[update]
            upper_bound = update_ix[update+1]
        else:
            raise("Not Implemented")
        if verbose:
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        for i in range(num_iters_per_update):
            if i==0:  # Then we reset our vars with the new update's data
                s_temp = full_trial_input_data[lower_bound:upper_bound]
                p_reference = torch.transpose(full_trial_labels[lower_bound:upper_bound], 0, 1)

                # First, normalize the entire s matrix
                if normalize_emg:
                    s_normed = s_temp / torch.linalg.norm(s_temp, ord='fro')
                    assert (torch.linalg.norm(s_normed, ord='fro')<1.2) and (torch.linalg.norm(s_normed, ord='fro')>0.8)
                else:
                    s_normed = s_temp
                # Apply PCA if applicable
                if PCA_comps!=64:  # 64 is the number of channels present on the recording armband  
                    pca = PCA(n_components=PCA_comps)
                    s = torch.transpose(torch.tensor(pca.fit_transform(s_normed), dtype=torch.float32), 0, 1)
                else:
                    s = torch.transpose(s_normed, 0, 1)
                    
                emg_streamed_batch = s[:,:-1] # F
                v_actual =  torch.matmul(model.weight, s)
                p_actual = torch.cumsum(v_actual, dim=1)*dt  # Numerical integration of v_actual to get p_actual
                V = (p_reference - p_actual)*dt
                if normalize_V:
                    V = V/torch.linalg.norm(V, ord='fro')
                    assert (torch.linalg.norm(V, ord='fro')<1.2) and (torch.linalg.norm(V, ord='fro')>0.8)

                y = p_reference[:, :-1]  # To match the input
                
                if verbose_norms:
                    print(f"Norm of Final s: {torch.linalg.norm(s, ord='fro')}")
                    print(f"Norm of emg_streamed_batch: {torch.linalg.norm(emg_streamed_batch, ord='fro')}")
                    print(f"Norm of D: {torch.linalg.norm(model.weight, ord='fro')}")
                    print(f"Norm of V: {torch.linalg.norm(V, ord='fro')}")
                    print()

            # TRAIN MODEL
            # reset gradient so it doesn't accumulate
            optimizer.zero_grad()
            # forward pass and loss
            y_pred = model(torch.transpose(emg_streamed_batch, 0, 1))  # Why do I have to transpose again here... my original code didn't
            # F, D, V, learning_batch
            if use_CPHSLoss:
                criterion = CPHSLoss
            else:
                criterion = cost_l2_torch
            loss = criterion(emg_streamed_batch, model.weight, V, emg_streamed_batch.shape[1], Ne=PCA_comps, lambdaF=lambdasFDE[0], lambdaD=lambdasFDE[1], lambdaE=lambdasFDE[2])  #, return_cost_func_comps=return_cost_func_comps)
            if return_cost_func_comps:
                #D = D.view(Nd, Ne)  #np.reshape(D,(Nd,Ne))             
                ETerm_log.append(lambdasFDE[2]*(torch.linalg.matrix_norm((torch.matmul(model.weight, emg_streamed_batch) - V[:,1:]))**2))
                DTerm_log.append(lambdasFDE[1]*(torch.linalg.matrix_norm((model.weight)**2)))
            # backward pass
            loss.backward(retain_graph=True)
            loss_log.append(loss.item())
            # update weights
            optimizer.step()
            
        if use_full_input_data:
            if verbose:
                print("Returning early!")
            # Eg we only go through 1 outer loop since we already looped through the full data
            if return_cost_func_comps:
                return model, loss_log, ETerm_log, DTerm_log
            else:
                return model, loss_log
    
    if return_cost_func_comps:
        return model, loss_log, ETerm_log, DTerm_log
    else:
        return model, loss_log