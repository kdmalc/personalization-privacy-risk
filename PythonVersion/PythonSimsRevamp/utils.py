import numpy as np
import matplotlib.pyplot as plt

def normalize_2D_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val == min_val:
        return np.zeros_like(array)
    
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

'''
# Example usage:
input_array = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=np.float32)

normalized_array = normalize_2D_array(input_array)
print(normalized_array)
'''


# Add this as a static method?
def condensed_external_plotting(input_data, version, exclusion_ID_lst=[], dim_reduc_factor=10, plot_gradient=False, global_error=True, local_error=True, pers_error=False, different_local_round_thresh_per_client=False, legend_on=False, plot_performance=False, plot_Dnorm=False, plot_Fnorm=False, num_participants=14, show_update_change=False, custom_title="", ylim=-1):
    id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
    
    global_alpha = 0.25
    global_linewidth = 3.5
    local_linewidth = 0.5
    pers_linewidth = 1
    
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
        
    running_max = 0
    for i in range(len(user_database)):
        # Skip over users that distort the scale
        if user_database[i].ID in exclusion_ID_lst:
            continue 
        else: 
            # This is used for plotting later
            if len(user_database[i].local_error_log) > running_max:
                running_max = len(user_database[i].local_error_log)

            if version.upper()=='LOCAL':
                if global_error:
                    df = pd.DataFrame(user_database[i].global_error_log)
                    # Need to append a col of indices for the dim reduc... just reset index?
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=global_linewidth, alpha=global_alpha)
                if local_error:
                    df = pd.DataFrame(user_database[i].local_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=local_linewidth)
                if pers_error:
                    df = pd.DataFrame(user_database[i].personalized_error_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle="--")
                # NOT THE COST FUNC, THESE ARE THE INDIVIDUAL COMPONENTS OF IT
                if plot_performance:
                    df = pd.DataFrame(user_database[i].performance_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, label=f"User{user_database[i].ID} Performance")
                if plot_Dnorm:
                    df = pd.DataFrame(user_database[i].Dnorm_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle="--", label=f"User{user_database[i].ID} Dnorm")
                if plot_Fnorm:
                    df = pd.DataFrame(user_database[i].Fnorm_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle=":", label=f"User{user_database[i].ID} Fnorm")
                if plot_gradient:
                    df = pd.DataFrame(user_database[i].gradient_log)
                    df.reset_index(inplace=True)
                    df10 = df.groupby(df.index//dim_reduc_factor, axis=0).mean()
                    plt.plot(df10.values[1:, 0], df10.values[1:, 1], color=id2color[user_database[i].ID], linewidth=2, label=f"User{user_database[i].ID} Gradient")
                #if different_local_round_thresh_per_client:
                #    print("DIFFERENT LOCAL THRESH FOR EACH CLIENT")
                #    if user_lst[i].data_stream == 'streaming':
                #        for my_update_transition in user_lst[i].update_transition_log:
                #            plt.scatter(my_update_transition+1, user_lst[i].local_error_log[my_update_transition]
                #                        [1], color=id2color[i], marker='*')
                #elif user_lst[i].ID==0 and show_update_change:  # Proxy for just printing it once...
                #    for thresh_idx in range(user_lst[i].current_threshold // user_lst[i].local_round_threshold):
                #        plt.axvline(x=(thresh_idx+1)*user_lst[i].local_round_threshold, color="k", linewidth=1, 
                #                    linestyle=':')
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
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=global_linewidth, alpha=global_alpha)

                if local_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.local_error_log[j][i][2])
                        client_global_round.append(input_data.local_error_log[j][i][1])
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=local_linewidth)
               
                if pers_error:
                    client_loss = []
                    client_global_round = []
                    for j in range(input_data.current_round):
                        client_loss.append(input_data.personalized_error_log[j][i][2])
                        client_global_round.append(input_data.personalized_error_log[j][i][1])
                    plt.plot(moving_average(client_global_round, dim_reduc_factor)[1:], moving_average(client_loss, dim_reduc_factor)[1:], color=id2color[user_database[i].ID], linewidth=pers_linewidth, linestyle="--")

                if show_update_change:
                    for update_round in user_database[i].update_transition_log:
                        plt.axvline(x=(update_round), color=id2color[user_database[i].ID], linewidth=0.5, alpha=0.6)  

    plt.ylabel('Cost L2')
    plt.xlabel('Iteration Number')
    plt.title(my_title)
    if version.upper()=='GLOBAL':
        running_max = input_data.current_round
    num_ticks = 5
    plt.xticks(ticks=np.linspace(0,running_max,num_ticks,dtype=int))
    plt.xlim((0,running_max+1))
    if ylim!=-1:
        plt.ylim((0,ylim))
    if legend_on:
        plt.legend()
    plt.show()
    
        
# Code for saving data needed for running sims
#cond0_dict_list = [0]*num_participants
#for idx in range(num_participants):
#    cond0_dict_list[idx] = {'training':emgs_block1[keys[idx]][0,:,:], 'labels':refs_block1[keys[idx]][0,:,:]}
#
#with open(path+cond0_filename, 'wb') as fp:
#    pickle.dump(cond0_dict_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
#    
#init_decoders = [Ws_block1[keys[i]][:, 0, :, :] for i in range(num_participants)]
#with open(path+all_decs_init_filename, 'wb') as fp:
#    pickle.dump(init_decoders, fp, protocol=pickle.HIGHEST_PROTOCOL)