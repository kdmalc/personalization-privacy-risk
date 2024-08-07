import numpy as np


path = r'C:\Users\kdmen\Desktop\Research\Data\CPHS_EMG'
cond0_filename = r'\cond0_dict_list.p'
all_decs_init_filename = r'\all_decs_init.p'
nofl_decs_filename = r'\nofl_decs.p'
id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
implemented_client_training_methods = ['GD', 'FullScipyMin', 'MaxIterScipyMin']
NUM_USERS = 14
# For exclusion when plotting later on
#bad_nodes = [] #[1,3,13]
D_0 = np.random.rand(2,64)
num_updates = 18
step_indices = list(range(num_updates))

STARTING_UPDATE=10
DATA_STREAM='streaming'
NUM_KFOLDS=5
USE_HITBOUNDS=False
PLOT_EACH_FOLD = False