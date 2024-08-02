import os
import numpy as np
np.random.seed(0)
import random
random.seed(0)

from matplotlib import pyplot as plt
#import seaborn as sns
#from presentation_sns_config import *

from experiment_params import *
from cost_funcs import *
from fl_sim_client import *
from fl_sim_server import *
import time
import pickle

path = r'C:\Users\kdmen\Desktop\Research\Data\CPHS_EMG'
cond0_filename = r'\cond0_dict_list.p'
all_decs_init_filename = r'\all_decs_init.p'
nofl_decs_filename = r'\nofl_decs.p'
id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
implemented_client_training_methods = ['GD', 'FullScipyMin', 'MaxIterScipyMin']
num_participants = 14

# For exclusion when plotting later on
bad_nodes = [] #[1,3,13]

with open(path+cond0_filename, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)
    
#with open(path+all_decs_init_filename, 'rb') as fp:
#    init_decoders = pickle.load(fp)    
#cond0_init_decs = [dec[0, :, :] for dec in init_decoders]

D_0 = np.random.rand(2,64)

num_updates = 18

step_indices = list(range(num_updates))

# CLIENT INIT FOR REFERNECE!
#def __init__(self, ID, w, opt_method, full_client_dataset, data_stream, smoothbatch=0.75, current_round=0, PCA_comps=64, 
#            availability=1, final_usable_update_ix=17, global_method='FedAvg', max_iter=1, normalize_EMG=True, starting_update=10, 
#            track_cost_components=True, gradient_clipping=False, log_decs=True, 
#            clipping_threshold=100, tol=1e-10, lr=1, track_gradient=True, wprev_global=False, 
#            num_steps=1, use_zvel=False, use_kfoldv=False, 
#            mix_in_each_steps=False, mix_mixed_SB=False, delay_scaling=0, random_delays=False, download_delay=1, 
#            upload_delay=1, copy_type='deep', validate_memory_IDs=True, local_round_threshold=50, condition_number=3, 
#            verbose=False, test_split_type='end', test_split_frac=0.3, use_up16_for_test=True)

# Need to be integrated!
# num_steps=1, use_zvel=False, use_kfoldv=False,
user_c0_1ScipyStep = [Client(i, copy.deepcopy(D_0), 'MaxiterScipyMin', cond0_training_and_labels_lst[i], 'streaming', starting_update=10, global_method='FedAvg', max_iter=1) for i in range(14)]
global_model_1scipystep = Server(1, copy.deepcopy(D_0), opt_method='MaxiterScipyMin', global_method='FedAvg', all_clients=user_c0_1ScipyStep)

big_loop_iters = 100
for i in range(big_loop_iters):
    if i%50==0:
        print(f"Round {i} of {big_loop_iters}")
    global_model_1scipystep.execute_FL_loop()
