import numpy as np
import h5py
import os


def load_final_model_performaces(cv_results_path, filename, type=None, h5_path=None, num_clients=14, num_folds=5, verbose=False):
    if h5_path is None:
        h5_path = os.path.join(cv_results_path, filename)
    if verbose:
        print(h5_path)
    
    # Load data from HDF5 file
    with h5py.File(h5_path, 'r') as f:
        a_group_key = list(f.keys())
        extraction_dict = dict()
        final_clients_loss_lst = []

        index = 0
        for key in a_group_key:
            extraction_dict[key] = f[key][()]
            if "client" in key:
                if isinstance(f[key][()], np.ndarray):
                    if len(extraction_dict[key])<2:
                        if verbose:
                            print(f"{index} {key}: len<2: appended -1 as placeholder!!!")
                        # This was a testing client and thus not trained...
                        final_clients_loss_lst.append(-1)
                        index += 1
                    else:
                        if verbose:
                            print(f"{index} {key}: ADDED!!!")
                        final_clients_loss_lst.append(f[key][()][-1])
                        index += 1
                else:
                    if verbose:
                        print(f"{key}: Not a numpy array!!!")
            else:
                if verbose:
                    print(f"{key}: Not a client log!!!")
    
    # Organize loss by client
    client_data_lst = [[] for _ in range(num_clients)]
    # Define the correct order mapping for the clients (client 0, 10, 11, 12, 13, 1, 2, ...)
    ## This is an artifact of the fact that the keys were strings and 10,11,etc comes before 1...
    correct_order = [0, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for trial_idx in range(num_folds):
        for client_idx in range(num_clients):
            # Use the correct client index mapping
            correct_client_idx = correct_order[client_idx]
            index = trial_idx * num_clients + client_idx
            client_data_lst[correct_client_idx].append(final_clients_loss_lst[index])
    
    if type=="CROSS":
        # Ensure that every client list has exactly one -1
        assert all(client_data.count(-1) == 1 for client_data in client_data_lst), "Some client lists do not have exactly one -1"

    # Remove all -1 values from each list
    cleaned_client_data_lst = [[x for x in sublist if x != -1] for sublist in client_data_lst]
    
    # Compute mean loss for each client
    client_final_loss_lst = [np.mean(np.array(cleaned_client_data_lst[client_idx])) for client_idx in range(num_clients)]
    
    if verbose:
        print("\n\n\n")
    
    return client_final_loss_lst

# INTRA
#intra_nofl = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_22-02_NOFL', 'FULLSCIPYMIN_NOFL_CrossValResults.h5')
#intra_fedavg = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_22-06_FEDAVG', 'GDLS_FEDAVG_CrossValResults.h5')
#intra_pfa1 = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_22-07_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5')
#intra_pfa2 = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_22-09_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5')
# CROSS
#cross_pfa = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_18-08_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5')
#cross_fedavg = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_18-07_FEDAVG', 'GDLS_FEDAVG_CrossValResults.h5')
#cross_nofl = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\Preliminary_Sims_Figure\08-21_16-24_NOFL', 'FULLSCIPYMIN_NOFL_CrossValResults.h5')

results_path = r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results'
current_directory = r'\Prelim_Sim_Results_V2'
base_path = results_path + current_directory

# CROSS
cross_pfa = load_final_model_performaces(base_path+r'\08-23_21-20_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5', type="CROSS")
cross_fedavg = load_final_model_performaces(base_path+r'\08-23_22-17_FEDAVG', 'GDLS_FEDAVG_CrossValResults.h5', type="CROSS")
cross_nofl = load_final_model_performaces(base_path+r'\08-23_22-18_NOFL', 'FULLSCIPYMIN_NOFL_CrossValResults.h5', type="CROSS")
#load_final_model_performaces(r'', filename)
# INTRA
intra_nofl = load_final_model_performaces(base_path+r'\08-23_20-43_NOFL_INTRA', 'FULLSCIPYMIN_NOFL_CrossValResults.h5', type="INTRA")
intra_fedavg = load_final_model_performaces(base_path+r'\08-23_20-42_FEDAVG_INTRA', 'GDLS_FEDAVG_CrossValResults.h5', type="INTRA")
intra_pfa1 = load_final_model_performaces(base_path+r'\08-23_18-56_PFAFO_GDLS_INTRA', 'GDLS_PFAFO_GDLS_CrossValResults.h5', type="INTRA")














