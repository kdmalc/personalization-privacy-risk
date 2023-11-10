import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import copy
from matplotlib import pyplot as plt

from experiment_params import *
from cost_funcs import *
import time
import pickle
from sklearn.decomposition import PCA


class ModelBase:
    # Hard coded attributes --> SHARED FOR THE ENTIRE CLASS
    # ^Are they? I'm not actually sure.  You can't access them obviously
    num_updates = 19
    cphs_starting_update = 10
    update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    pers_methods = ['FedAvgSB', 'APFL', 'Per-FedAvg FO', 'Per-FedAvg HF']
    
    def __init__(self, ID, w, method, smoothbatch=1, verbose=False, PCA_comps=7, current_round=0, num_participants=14, log_init=0):
        self.type = 'BaseClass'
        self.ID = ID
        self.PCA_comps = PCA_comps
        self.pca_channel_default = 64  # When PCA_comps equals this, DONT DO PCA
        if w.shape!=(2, self.PCA_comps):
            #print(f"Class BaseModel: Overwrote the provided init decoder: {w.shape} --> {(2, self.PCA_comps)}")
            self.w = np.random.rand(2, self.PCA_comps)
        else:
            self.w = w
        self.w_prev = copy.deepcopy(self.w)
        self.dec_log = [copy.deepcopy(self.w)]
        self.w_prev = copy.deepcopy(self.w)
        self.num_participants = num_participants
        self.log_init = log_init
        self.local_error_log = [] #[log_init]*num_participants
        self.global_error_log = [] #[log_init]*num_participants
        self.pers_error_log = [] #[log_init]*num_participants
        self.local_test_error_log = [] #[log_init]*num_participants
        self.global_test_error_log = [] #[log_init]*num_participants
        self.pers_test_error_log = [] #[log_init]*num_participants
        self.method = method
        self.current_round = current_round
        self.verbose = verbose
        self.smoothbatch = smoothbatch

        
    def __repr__(self): 
        return f"{self.type}{self.ID}"
    
    def display_info(self): 
        return f"{self.type} model: {self.ID}\nCurrent Round: {self.current_round}\nTraining Method: {self.method}"


class TrainingMethods:
    # Different training approaches
    
    # This one blows up to NAN/overflow... not sure why
    def train_eta_gradstep(self, w, eta, F, D, H, V, learning_batch, alphaF, alphaD, PCA_comps):
        grad_cost = np.reshape(gradient_cost_l2(F, D, H, V, learning_batch, alphaF, alphaD, Ne=PCA_comps),(2, PCA_comps))
        w_new = w - eta*grad_cost
        return w_new

    def train_eta_scipyminstep(self, w, eta, F, D, H, V, learning_batch, 
                               alphaF, alphaD, D0, display_info, PCA_comps, full=False):
        '''
        full: when True, will run a complete minimization. When False, will only run eta gradient steps (ie 1).
        '''
        
        if full:
            out = minimize(
                lambda D: cost_l2(F,D,H,V,learning_batch, alphaF,alphaD,Ne=PCA_comps), D0, method='BFGS', 
                jac=lambda D: gradient_cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps))#, options={'disp': display_info})
        else:
            out = minimize(
                lambda D: cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps), D0, method='BFGS', 
                jac=lambda D: gradient_cost_l2(F,D,H,V,learning_batch,alphaF,alphaD,Ne=PCA_comps), 
                options={'maxiter':eta}) #'disp': display_info, 
        w_new = np.reshape(out.x,(2, PCA_comps))
        return w_new


# Add this as a static method?
def condensed_external_plotting(input_data, version, exclusion_ID_lst=[], dim_reduc_factor=1, plot_gradient=False, plot_pers_gradient=False, plot_this_ID_only=-1, plot_global_gradient=False, global_error=True, local_error=True, pers_error=False, different_local_round_thresh_per_client=False, legend_on=False, plot_performance=False, plot_Dnorm=False, plot_Fnorm=False, num_participants=14, show_update_change=True, custom_title="", axes_off_list=[], ylim_max=None, ylim_min=None, my_legend_loc='best', global_alpha=1, local_alpha=1, pers_alpha=1, global_linewidth=1, local_linewidth=1, pers_linewidth=1, global_linestyle='dashed', local_linestyle='solid', pers_linestyle='dotted'):
    
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    
    def moving_average(numbers, window_size):
        i = 0
        moving_averages = []
        while i < len(numbers) - window_size + 1:
            this_window = numbers[i : i + window_size]

            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += window_size
        return moving_averages
    
    if custom_title:
        my_title = custom_title
    elif global_error and local_error:
        my_title = f'Global and Local Costs Per {version.title()} Iter'
    elif global_error:
        my_title = f'Global Cost Per {version.title()} Iter'
    elif local_error:
        my_title = f'Local Costs Per {version.title()} Iter'
    else:
        raise ValueError("You set both global and local to False.  At least one must be true in order to plot something.")

    # Determine if this is global or local, based on the input for now... could probably add a flag but meh
    if version.upper()=='LOCAL':
        user_database = input_data
    elif version.upper()=='GLOBAL':
        user_database = input_data.all_clients
    else:
        raise ValueError("log_type must be either global or local, please retry")
        
    max_local_iters = 0

    for i in range(len(user_database)):
        # Skip over users that distort the scale
        if user_database[i].ID in exclusion_ID_lst:
            continue 
        elif plot_this_ID_only!=-1 and i!=plot_this_ID_only:
            continue
        elif len(user_database[i].local_error_log)<2:
            # This node never trained so just skip it so it doesn't break the plotting
            continue 
        else: 
            # This is used for plotting later
            if len(user_database[i].local_error_log) > max_local_iters:
                max_local_iters = len(user_database[i].local_error_log)

            if version.upper()=='LOCAL':
                if global_error:
                    df = pd.DataFrame(user_database[i].global_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=global_linewidth, alpha=global_alpha, linestyle=global_linestyle)
                if local_error:
                    df = pd.DataFrame(user_database[i].local_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=local_linewidth, alpha=local_alpha, linestyle=local_linestyle)
                if pers_error:
                    df = pd.DataFrame(user_database[i].pers_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, alpha=pers_alpha, linestyle=pers_linestyle)
                # NOT THE COST FUNC, THESE ARE THE INDIVIDUAL COMPONENTS OF IT
                if plot_performance:
                    df = pd.DataFrame(user_database[i].performance_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, label=f"User{user_database[i].ID} Performance")
                if plot_Dnorm:
                    df = pd.DataFrame(user_database[i].Dnorm_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle="--", label=f"User{user_database[i].ID} Dnorm")
                if plot_Fnorm:
                    df = pd.DataFrame(user_database[i].Fnorm_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle=":", label=f"User{user_database[i].ID} Fnorm")
                if plot_gradient:
                    df = pd.DataFrame(user_database[i].gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Local Gradient")
                if plot_pers_gradient:
                    df = pd.DataFrame(user_database[i].pers_gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Pers Gradient")
                if plot_global_gradient:
                    df = pd.DataFrame(user_database[i].global_gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[:, 0], df10.values[:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Global Gradient")
            elif version.upper()=='GLOBAL':
                if plot_Fnorm or plot_Dnorm or plot_performance:
                    print("Fnorm, Dnorm, and performance are currently not supported for version==GLOBAL")
                    
                if global_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.global_error_log[j][i][2])
                        # This is actually the client local round
                        client_global_round.append(input_data.global_error_log[j][i][1])
                    # Why is the [1:] here?  What happens when dim_reduc=1? 
                    # Verify that this is the same as my envelope code...
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=global_linewidth, alpha=global_alpha, linestyle=global_linestyle)

                if local_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.local_error_log[j][i][2])
                        client_global_round.append(input_data.local_error_log[j][i][1])
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=local_linewidth, alpha=local_alpha, linestyle=local_linestyle)
               
                if pers_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.pers_error_log[j][i][2])
                        client_global_round.append(input_data.pers_error_log[j][i][1])
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=pers_linewidth, alpha=pers_alpha, linestyle=pers_linestyle)

                if show_update_change:
                    for update_round in user_database[i].update_transition_log:
                        plt.axvline(x=(update_round), color=id2color[user_database[i].ID], linewidth=0.5, alpha=0.6)  

    if version.upper()=='LOCAL' and show_update_change==True:
        for i in range(max_local_iters):
            if i%user_database[0].local_round_threshold==0:
                plt.axvline(x=i, color="k", linewidth=1, linestyle=':')
                
    if axes_off_list!=[]:
        ax = plt.gca()
        for my_axis in axes_off_list:
            ax.spines[my_axis].set_visible(False)
              
    plt.ylabel('Cost L2')
    plt.xlabel('Iteration Number')
    plt.title(my_title)
    if version.upper()=='GLOBAL':
        max_local_iters = input_data.current_round
    else:
        num_ticks = 5
        plt.xticks(ticks=np.linspace(0,max_local_iters,num_ticks,dtype=int))
        plt.xlim((0,max_local_iters+1))
    if ylim_max!=None:
        if ylim_min!=None:
            plt.ylim((ylim_min,ylim_max))
        else:
            plt.ylim((0,ylim_max))
    if legend_on:
        plt.legend(loc=my_legend_loc)
    plt.show()
    

def central_tendency_plotting(all_user_input, highlight_default=False, default_local=False, default_global=False, default_pers=False, plot_mean=True, plot_median=False, exclusion_ID_lst=[], dim_reduc_factor=1, plot_gradient=False, plot_pers_gradient=False, plot_this_ID_only=-1, plot_global_gradient=False, global_error=True, local_error=True, pers_error=False, different_local_round_thresh_per_client=False, legend_on=True, plot_performance=False, plot_Dnorm=False, plot_Fnorm=False, num_participants=14, show_update_change=True, custom_title="", axes_off_list=[], xlim_max=None, xlim_min=None, ylim_max=None, ylim_min=None, input_linewidth=1, my_legend_loc='best', iterable_labels=[], iterable_colors=[]):
    
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    
    num_central_tendencies = 2  # Mean and median... idk, maybe use flags or something...
    
    if dim_reduc_factor!=1:
        raise("dim_reduc_factor MUST EQUAL 1!")
        
    global_df = pd.DataFrame()
    local_df = pd.DataFrame()
    pers_df = pd.DataFrame()
    perf_df = pd.DataFrame()
    dnorm_df = pd.DataFrame()
    fnorm_df = pd.DataFrame()
    grad_df = pd.DataFrame()
    pers_grad_df = pd.DataFrame()
    global_grad_df = pd.DataFrame()
    
    param_list = [plot_gradient, plot_pers_gradient, plot_global_gradient, global_error, local_error, pers_error, plot_performance, plot_Dnorm, plot_Fnorm]
    all_vecs_dict = dict()
    all_vecX_dict = dict()
    for param_idx, param in enumerate(param_list):
        all_vecs_dict[param_idx] = [[] for _ in range(num_central_tendencies)]
        all_vecX_dict[param_idx] = [[] for _ in range(num_central_tendencies)]
    param_label_dict = {0:'Gradient', 1:'Personalized Gradient', 2:'Global Gradient', 3:'Global Error', 4:'Local Error', 5:'Personalized Error', 6:'Performance', 7:'DNorm', 8:'FNorm'}
    tendency_label_dict = {0:'Mean', 1:'Pseudo-Median'}
    
    def moving_average(numbers, window_size):
        i = 0
        moving_averages = []
        while i < len(numbers) - window_size + 1:
            this_window = numbers[i : i + window_size]

            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += window_size
        return moving_averages
    
    if custom_title:
        my_title = custom_title
    elif global_error and local_error:
        my_title = 'Global and Local Costs Per Local Iter'
    elif global_error:
        my_title = 'Global Cost Per Local Iter'
    elif local_error:
        my_title = 'Local Costs Per Local Iter'
    else:
        print("FYI You set both global and local to False.  No title set.")
        my_title = 'Please enter a custom title'
        

    max_local_iters = 0
    label_idx = 0
    for user_idx, user_database in enumerate(all_user_input):
        for i in range(len(user_database)):
            # Skip over users that distort the scale
            if user_database[i].ID in exclusion_ID_lst:
                continue 
            elif len(user_database[i].local_error_log)<2:
                # This node never trained so just skip it so it doesn't break the plotting
                continue 
            else: 
                # This is used for plotting later
                if len(user_database[i].local_error_log) > max_local_iters:
                    max_local_iters = len(user_database[i].local_error_log)

                # This is how it would be supposed to work
                # Append needs to change to concat
                # ISSUE: I loop through the iters and so the dfs aren't actually built yet
                # So I would have to build each df (do concat and have some base init...)
                #for flag_idx, plotting_flag in enumerate(param_list):
                #    if plotting_flag:
                #        df = pd.DataFrame(user_database[i].global_error_log)
                #         df.reset_index(inplace=True)
                #        global_df.append(df.groupby(df.index//dim_reduc_factor, axis=0).mean())
                #        all_dfs_dict[flag_idx]
                #        for column in my_df:
                #            if 'MEAN' in central_tendency.upper():
                    #            all_vecs_dict[flag_idx].append(pd.DataFrame(my_df[column].tolist().mean().tolist()))
                #            if 'MEDIAN' in central_tendency.upper():
                #                all_vecs_dict[flag_idx].append(pd.DataFrame(my_df.median(axis=0)))
                if global_error or (user_idx==0 and default_global==True):
                    df = pd.DataFrame(user_database[i].global_error_log)
                    global_df = pd.concat([global_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if local_error or (user_idx==0 and default_local==True):
                    df = pd.DataFrame(user_database[i].local_error_log)
                    local_df = pd.concat([local_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if pers_error or (user_idx==0 and default_pers==True):
                    df = pd.DataFrame(user_database[i].pers_error_log)
                    pers_df = pd.concat([pers_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_performance:
                    df = pd.DataFrame(user_database[i].performance_log)
                    perf_df = pd.concat([perf_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_Dnorm:
                    df = pd.DataFrame(user_database[i].Dnorm_log)
                    dnorm_df = pd.concat([dnorm_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_Fnorm:
                    df = pd.DataFrame(user_database[i].Fnorm_log)
                    fnorm_df = pd.concat([fnorm_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_gradient:
                    df = pd.DataFrame(user_database[i].gradient_log)
                    grad_df = pd.concat([grad_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_pers_gradient:
                    df = pd.DataFrame(user_database[i].pers_gradient_log)
                    pers_grad_df = pd.concat([pers_grad_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])
                if plot_global_gradient:
                    df = pd.DataFrame(user_database[i].global_gradient_log)
                    global_grad_df = pd.concat([global_grad_df, (df.groupby(df.index//dim_reduc_factor, axis=0).mean()).T])

        # Bad temporary soln for MVP
        all_dfs_dict = {0:grad_df.reset_index(drop=True), 1:pers_grad_df.reset_index(drop=True), 2:global_grad_df.reset_index(drop=True), 3:global_df.reset_index(drop=True), 4:local_df.reset_index(drop=True), 5:pers_df.reset_index(drop=True), 6:perf_df.reset_index(drop=True), 7:dnorm_df.reset_index(drop=True), 8:fnorm_df.reset_index(drop=True)}
        
        for flag_idx, plotting_flag in enumerate(param_list):
            if plotting_flag:
                my_df = all_dfs_dict[flag_idx]
                if plot_mean:
                    all_vecs_dict[flag_idx][0] = my_df.mean()
                if plot_median:
                    all_vecs_dict[flag_idx][1] = my_df.median()

        if show_update_change==True:
            for i in range(max_local_iters):
                if i%user_database[0].local_round_threshold==0:
                    plt.axvline(x=i, color="k", linewidth=1, linestyle=':') 

        for flag_idx, plotting_flag in enumerate(param_list):
            if plotting_flag:
                my_vec = all_vecs_dict[flag_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        if iterable_labels!=[]:
                            my_label = iterable_labels[label_idx]
                            label_idx += 1
                        else:
                            my_label = f"{tendency_label_dict[vec_idx]} {param_label_dict[flag_idx]}"
                        if "GLOBAL:" in my_label.upper():
                            my_linestyle = 'dashed'
                        elif "LOCAL:" in my_label.upper():
                            my_linestyle = 'solid'
                        elif "PERS:" in my_label.upper():
                            my_linestyle = 'dotted'
                        else:
                            my_linestyle = 'solid'
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth, linestyle=my_linestyle)
                        
                        
        #param_list FOR REFERENCE: [plot_gradient, plot_pers_gradient, plot_global_gradient, global_error, local_error, pers_error, plot_performance, plot_Dnorm, plot_Fnorm]         
        if user_idx==0:
            if default_global:  # 3 corresponds to global
                global_idx = 3
                all_vecs_dict[global_idx][0] = all_dfs_dict[global_idx].mean()
                all_vecs_dict[global_idx][1] = all_dfs_dict[global_idx].median()
                my_vec = all_vecs_dict[global_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        my_label = f"{tendency_label_dict[vec_idx]} Global Error"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
            if default_local:  # 4 corresponds to local
                local_idx = 4
                all_vecs_dict[local_idx][0] = all_dfs_dict[local_idx].mean()
                all_vecs_dict[local_idx][1] = all_dfs_dict[local_idx].median()
                my_vec = all_vecs_dict[local_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        my_label = f"{tendency_label_dict[vec_idx]} Local Error"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
            if default_pers:  # 5 corresponds to pers
                pers_idx = 5
                all_vecs_dict[pers_idx][0] = all_dfs_dict[pers_idx].mean()
                all_vecs_dict[pers_idx][1] = all_dfs_dict[pers_idx].median()
                my_vec = all_vecs_dict[pers_idx]
                for vec_idx, vec_vec in enumerate(my_vec):
                    if (plot_mean==True and vec_idx==0) or (plot_median==True and vec_idx==1):
                        my_label = f"{tendency_label_dict[vec_idx]} Personalized Error"
                        my_alpha = 0.4 if (highlight_default and user_idx==0) else 1
                        my_linewidth = 5 if (highlight_default and user_idx==0) else input_linewidth
                        plt.plot(range(len(vec_vec)), vec_vec, label=my_label, alpha=my_alpha, linewidth=my_linewidth)
    
    plt.ylabel('Cost L2')
    plt.xlabel('Iteration Number')
    plt.title(my_title)
    num_ticks = 5
    plt.xticks(ticks=np.linspace(0,max_local_iters,num_ticks,dtype=int))
    plt.xlim((0,max_local_iters+1))
    
    if ylim_max!=None:
        if ylim_min!=None:
            plt.ylim((ylim_min,ylim_max))
        else:
            plt.ylim((0,ylim_max))
    if xlim_max!=None:
        if xlim_min!=None:
            plt.xlim((xlim_min,xlim_max))
        else:
            plt.xlim((0,xlim_max))
            
    if legend_on:
        plt.legend(loc=my_legend_loc)
    
    if axes_off_list!=[]:
        # left bottom top right 1 1 1 1
        ax = plt.gca()
        for key_pair in axes_off_list:
            ax.spines[key_pair[0]].set_visible(key_pair[1])
        
    plt.show()
    
    return user_database, all_dfs_dict, all_vecs_dict
    
    
##############################################################################
# Zero vel boundary code:
def reconstruct_trial_fixed_decoder(ref_tr, emg_tr, Ds_fixed, time_x, fs = 60):
    time_x = time_x
    vel_est = np.zeros_like((ref_tr))
    pos_est = np.zeros_like((ref_tr))
    int_vel_est = np.zeros_like((ref_tr))

    hit_bound = 0
    vel_est[0] = Ds_fixed@emg_tr[0]  # D@s --> Kai's v_actual
    pos_est[0] = [0, 0]
    for tt in range(1, time_x):
        vel_plus = Ds_fixed@emg_tr[tt] # at time tt --> also Kai's v_actual...
        p_plus = pos_est[tt-1, :] + (vel_est[tt-1, :]/fs)
        # These are just correctives, such that vel_plus can get bounded
        # x-coordinate
        if abs(p_plus[0]) > 36:
            p_plus[0] = pos_est[tt-1, 0]
            vel_plus[0] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter
        if abs(p_plus[1]) > 24:
            p_plus[1] = pos_est[tt-1, 1]
            vel_plus[1] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter
        if hit_bound > 200:
            p_plus[0] = 0
            vel_plus[0] = 0
            p_plus[1] = 0
            vel_plus[1] = 0
            hit_bound = 0
        # now update velocity and position
        vel_est[tt] = vel_plus
        pos_est[tt] = p_plus
        # calculate intended velocity
        int_vel_est[tt] = calculate_intended_vels(ref_tr[tt], p_plus, 60)
    return vel_est, pos_est, int_vel_est


def calculate_intended_vels(ref, pos, fs):
    '''
    ref = 1 x 2
    pos = 1 x 2
    fs = scalar
    '''
    
    gain = 120
    ALMOST_ZERO_TOL = 0.01
    intended_vector = (ref - pos)/fs
    if np.linalg.norm(intended_vector) <= ALMOST_ZERO_TOL:
        intended_norm = np.zeros((2,))
    else:
        intended_norm = intended_vector * gain
    return intended_norm