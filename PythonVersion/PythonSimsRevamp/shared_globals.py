import numpy as np
from datetime import datetime


path = r'C:\Users\kdmen\Desktop\Research\Data\CPHS_EMG'
model_saving_dir = r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\PythonVersion\PythonSimsRevamp\models"
cond0_filename = r'\cond0_dict_list.p'
all_decs_init_filename = r'\all_decs_init.p'
nofl_decs_filename = r'\nofl_decs.p'
id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
implemented_client_training_methods = ['GD', 'FullScipyMin', 'MaxIterScipyMin']
NUM_USERS = 14
D_0 = np.random.rand(2,64)
num_updates = 18  # Is this even used anymore...
step_indices = list(range(num_updates))

MAX_ITER=None  # For MAXITERSCIPYMIN. Use FULLSCIPYMIN for complete minimization, otherwise stay with 1

COLORS_LST = ['red', 'blue', 'magenta', 'orange', 'darkviolet', 'lime', 'cyan', 'yellow']
ALPHA = 0.7

# get current date and time
CURRENT_DATETIME = str(datetime.now().strftime("%m-%d_%H-%M"))

STARTING_UPDATE=10
DATA_STREAM='streaming'
NUM_KFOLDS=7
USE_HITBOUNDS = False
PLOT_EACH_FOLD = False
USE_KFOLDCV = True
TEST_SPLIT_TYPE = 'KFOLDCV'