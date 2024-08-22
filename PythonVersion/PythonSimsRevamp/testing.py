import numpy as np
import h5py
import torch
import os
import matplotlib.pyplot as plt


def load_final_model_performaces(cv_results_path, filename, h5_path=None, num_clients=14, num_folds=5):
    if h5_path is None:
        h5_path = os.path.join(cv_results_path, filename)
    
    # Load data from HDF5 file
    with h5py.File(h5_path, 'r') as f:
        a_group_key = list(f.keys())
        extraction_dict = dict()
        final_clients_loss_lst = []

        for key in a_group_key:
            extraction_dict[key] = f[key][()]
            if isinstance(f[key][()], np.ndarray):
                final_clients_loss_lst.append(f[key][()][-1])
                #print(key)

        
    
    # Organize loss by client
    client_data_lst = [[] for _ in range(num_clients)]
    
    for trial_idx in range(num_folds):
        for client_idx in range(num_clients):
            index = trial_idx * num_clients + client_idx
            client_data_lst[client_idx].append(final_clients_loss_lst[index])
    
    # Compute mean loss for each client
    client_final_loss_lst = [np.mean(np.array(client_data_lst[client_idx])) for client_idx in range(num_clients)]
    
    # Print results
    #for i, client in enumerate(client_data_lst):
    #    print(f"Client {i+1}: {client}")
    #print("Client final loss list:", client_final_loss_lst)
    
    return client_final_loss_lst


# INTRA
intra_nofl = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_22-02_NOFL', 'FULLSCIPYMIN_NOFL_CrossValResults.h5')
intra_fedavg = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_22-06_FEDAVG', 'GDLS_FEDAVG_CrossValResults.h5')
intra_pfa1 = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_22-07_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5')
intra_pfa2 = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_22-09_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5')

# CROSS
cross_pfa = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_18-08_PFAFO_GDLS', 'GDLS_PFAFO_GDLS_CrossValResults.h5')
cross_fedavg = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_18-07_FEDAVG', 'GDLS_FEDAVG_CrossValResults.h5')
cross_nofl = load_final_model_performaces(r'C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\results\08-21_16-24_NOFL', 'FULLSCIPYMIN_NOFL_CrossValResults.h5')
#load_final_model_performaces(r'', filename)














