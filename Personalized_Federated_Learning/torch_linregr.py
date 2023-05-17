import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

def cost_l2_torch(F, D, V, learning_batch, lambdaF=1e-7, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64):
    # c_L2 = (lambdaE||DF + V+||_2)^2 + lambdaD*(||D||_2)^2 + lambdaF*(||F||_2)^2
    
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

def full_train_linregr_updates(model, full_trial_input_data, full_trial_labels, learning_rate, normalize_emg=False, PCA_comps=64, num_iters_per_update=30, starting_update=10, use_full_data=False, dt=1/60, loss_log=None, update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]):
    
    if loss_log is None:
        loss_log = list()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
    for update in range(len(update_ix[starting_update:])-1):
        update += starting_update
        
        if use_full_data:
            lower_bound = update_ix[starting_update]
            upper_bound = update_ix[-1]
        else:
            lower_bound = update_ix[update]
            upper_bound = update_ix[update+1]

        for i in range(num_iters_per_update):
            if i==0:  # Then we reset our vars with the new update's data
                s_temp = full_trial_input_data[lower_bound:upper_bound]
                p_reference = torch.transpose(full_trial_labels[lower_bound:upper_bound], 0, 1)

                # First, normalize the entire s matrix
                if normalize_emg:
                    s_normed = s_temp/torch.max(s_temp)
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

                y = p_reference[:, :-1]  # To match the input

            # TRAIN MODEL
            # reset gradient so it doesn't accumulate
            optimizer.zero_grad()
            # forward pass and loss
            y_pred = model(torch.transpose(emg_streamed_batch, 0, 1))  # Why do I have to transpose again here... my original code didn't
            # F, D, V, learning_batch
            loss = cost_l2_torch(emg_streamed_batch, model.weight, V, emg_streamed_batch.shape[1], Ne=PCA_comps)
            # backward pass
            loss.backward(retain_graph=True)
            loss_log.append(loss.item())
            # update weights
            optimizer.step()
            
        if use_full_data:
            # Eg we only go through 1 outer loop since we already looped through the full data
            return model, loss_log
            
    return model, loss_log