import pandas as pd
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

from itertools import permutations

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

user_c0_1ScipyStep = [Client(i, copy.deepcopy(D_0), 'MaxiterScipyMin', cond0_training_and_labels_lst[i], 'streaming', starting_update=10) for i in range(14)]
global_model_1scipystep = Server(1, copy.deepcopy(D_0), opt_method='MaxiterScipyMin', global_method='FedAvg', all_clients=user_c0_1ScipyStep)

big_loop_iters = 100
for i in range(big_loop_iters):
    if i%50==0:
        print(f"Round {i} of {big_loop_iters}")
    global_model_1scipystep.execute_FL_loop()
