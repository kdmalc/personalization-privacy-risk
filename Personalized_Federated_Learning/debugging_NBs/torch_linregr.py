import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from utils.custom_loss_class import CPHSLoss
from utils.custom_loss_class2 import CPHSLoss2


def cost_l2_torch(F, D, V, learning_batch, model=None, use_model_outputs=False, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False):
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
    if use_model_outputs:
        if model==None:
            raise ValueError("You must pass in a model object. model=my_model")
        else:
            v_pred = model(F)
    else:
        v_pred = torch.matmul(D, F)
    term1 = lambdaE*(torch.linalg.matrix_norm((v_pred - Vplus))**2)
    # D Norm
    term2 = lambdaD*(torch.linalg.matrix_norm((D))**2)
    # F Norm
    term3 = lambdaF*(torch.linalg.matrix_norm((F))**2)
    return (term1 + term2 + term3)

def full_train_linregr_updates(model, full_trial_input_data, full_trial_labels, learning_rate, lambdasFDE=[0, 1e-3, 1e-6], use_CPHSLoss=False, use_CPHSLoss2=False, normalize_data=False, PCA_comps=64, num_iters_per_update=30, starting_update=10, use_full_input_data=False, stream_data_updates=True, use_model_outputs=False, compare_model_and_matmul=False, dt=1/60, loss_log=None, verbose=False, verbose_norms=False, return_cost_func_comps=False, update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]):
    print("full_train_linregr_updates Params:")
    print(f"learning_rate {learning_rate}")
    print(f"lambdasFDE {lambdasFDE}")
    print(f"use_CPHSLoss {use_CPHSLoss}; use_CPHSLoss2 {use_CPHSLoss2}")
    print(f"normalize_emg {normalize_data}")
    print(f"num_iters_per_update {num_iters_per_update}")
    print(f"starting_update {starting_update}")
    print(f"use_full_input_data {use_full_input_data}")
    print(f"stream_data_updates {stream_data_updates}")
    print(f"return_cost_func_comps {return_cost_func_comps}")   
    
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
        #num_updates = len(update_ix[starting_update:])-1-1
        #-1 to account for the update+1 we do for upper bound
        #-1 again to account for us wanting to skip the last update since it is truncated
        
        # Take amount of input data and divide by 1200 AKA roughly the length of an update. Convert to int
        num_updates = int(np.floor(full_trial_input_data[starting_update:, :].shape[0] / 1200))
    if verbose:
        print(f"Num_updates: {num_updates}")
    
    print_switch = 1
        
    for update in range(starting_update, num_updates):  
        if verbose:
            print(f"Current Update: {update}")
        
        if use_full_input_data:
            lower_bound = 0
            upper_bound = -1
        elif stream_data_updates:
            lower_bound = update_ix[update]
            upper_bound = update_ix[update+1]
        else:
            raise("Not Implemented")
        if verbose:
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        for i in range(num_iters_per_update):
            if i==0:  # Then we reset our vars with the new update's data
                s_temp = full_trial_input_data[lower_bound:upper_bound, :]
                #print(f"s_temp shape: {s_temp.shape}")
                p_reference = torch.transpose(full_trial_labels[lower_bound:upper_bound, :], 0, 1)
                #print(f"p_reference shape: {p_reference.shape}")

                # First, normalize the entire s matrix
                if normalize_data:
                    s_normed = s_temp / torch.linalg.norm(s_temp, ord='fro')
                    assert (torch.linalg.norm(s_normed)<1.2) and (torch.linalg.norm(s_normed)>0.8)
                    p_reference = p_reference/torch.linalg.norm(p_reference)
                    assert (torch.linalg.norm(p_reference)<1.2) and (torch.linalg.norm(p_reference)>0.8)
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
                #print(f"y shape: {y.shape}")
                
                if compare_model_and_matmul:
                    v_pred_model = model(torch.transpose(emg_streamed_batch,0,1))
                    v_pred_matmul = torch.transpose(torch.matmul(model.weight, emg_streamed_batch),0,1)
                    v_pred_matmul_s = torch.transpose(torch.matmul(model.weight, s),0,1)
                    # What's the difference between v_actual (D@s) and D@F?
                    print()
                    print(f"v_pred_model shape: {v_pred_model.shape};\n vals[:5,:]: {v_pred_model[:5,:]};\n vals[-5:,:]: {v_pred_model[-5:,:]}")
                    print("---------------------------------------")
                    print(f"v_pred_matmul shape: {v_pred_matmul.shape};\n vals[:5,:]: {v_pred_matmul[:5,:]};\n vals[-5:,:]: {v_pred_matmul[-5:,:]}")
                    print("---------------------------------------")
                    print(f"v_pred_matmul_s shape: {v_pred_matmul_s.shape};\n vals[:5,:]: {v_pred_matmul_s[:5,:]};\n vals[-6:,:]: {v_pred_matmul_s[-6:,:]}")
                    print()
                
                if verbose==True or verbose_norms==True:
                    print(f"Norm of Final s: {torch.linalg.norm(s, ord='fro')}")
                    print(f"Norm of emg_streamed_batch: {torch.linalg.norm(emg_streamed_batch, ord='fro')}")
                    print(f"Norm of D: {torch.linalg.norm(model.weight, ord='fro')}")
                    print(f"Norm of V: {torch.linalg.norm(V, ord='fro')}")
                    print()

            # TRAIN MODEL
            # reset gradient so it doesn't accumulate
            optimizer.zero_grad()
            # forward pass and loss
            ## Uhhh this is my "output" right? Idk where the output that I was using below is even defined lmao
            y_pred = model(torch.transpose(emg_streamed_batch, 0, 1))  # Why do I have to transpose again here... my original code didn't
            #print(f"y_pred shape: {y_pred.shape}")
            # F, D, V, learning_batch
            if use_CPHSLoss:
                criterion = CPHSLoss
                # Here I am initializing CPHLoss since it takes an input
                ## Uh should it be model.weight or just modeL? I think just model...
                loss_func = criterion(emg_streamed_batch, model, V, emg_streamed_batch.shape[1], Ne=PCA_comps, lambdaF=lambdasFDE[0], lambdaD=lambdasFDE[1], lambdaE=lambdasFDE[2])  #, return_cost_func_comps=return_cost_func_comps)        
                
                if y_pred.shape[0]!=y.shape[0]:
                    #print()
                    #print(f"output shape: {y_pred.shape}")
                    #print(f"y shape: {y.shape}")
                    #print("Transposed!")
                    #print()
                    ty_pred = torch.transpose(y_pred, 0, 1)
                else:
                    ty_pred = y_pred
                loss = loss_func(ty_pred, y, model)
            elif use_CPHSLoss2:
                loss_func = CPHSLoss2(lambdaF=lambdasFDE[0], lambdaD=lambdasFDE[1], lambdaE=lambdasFDE[2])
                if y_pred.shape[0]!=y.shape[0]:
                    ty_pred = torch.transpose(y_pred, 0, 1)
                else:
                    ty_pred = y_pred
                t2 = lambdasFDE[1]*(torch.linalg.matrix_norm((model.weight))**2)
                t3 = lambdasFDE[0]*(torch.linalg.matrix_norm((emg_streamed_batch))**2)
                loss = loss_func(ty_pred, y) + t2 + t3
            else:
                criterion = cost_l2_torch
                # model.weight is actually correct for cost_l2_torch
                loss = criterion(emg_streamed_batch, model.weight, V, emg_streamed_batch.shape[1], use_model_outputs=use_model_outputs, Ne=PCA_comps, lambdaF=lambdasFDE[0], lambdaD=lambdasFDE[1], lambdaE=lambdasFDE[2])  #, return_cost_func_comps=return_cost_func_comps)
            
            if return_cost_func_comps:
                #D = D.view(Nd, Ne)  #np.reshape(D,(Nd,Ne))             
                ETerm_log.append(lambdasFDE[2]*(torch.linalg.matrix_norm((torch.matmul(model.weight, emg_streamed_batch) - V[:,1:]))**2))
                DTerm_log.append(lambdasFDE[1]*(torch.linalg.matrix_norm((model.weight))**2))
            
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