# PFLNIID

import copy
import torch
import torch.nn as nn
import numpy as np
#import random
import os
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from utils.processing_funcs import normalize_tensor
from utils.custom_loss_class import CPHSLoss
from utils.emg_dataset_class import *

#https://www.youtube.com/watch?v=3GVUzwXXihs
#^ Very helpful video about samplers and getting data from DataLoaders
# We will just stick with sequential, which he doesn't show, but it is obvious how it works after seeing the vid


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, ID, samples_path, labels_path, condition_number=None, **kwargs):
        # Why do I even have this take any arguments instead of just using args...

        self.model = copy.deepcopy(args.model)
        self.model_str = args.model_str
        if self.model_str != "LinearRegression":
            self.deep_bool = True
        else:
            self.deep_bool = False
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.ID = ID  # integer for now... maybe switch to subject codes later?
        self.save_folder_name = args.save_folder_name

        self.samples_path = samples_path
        self.labels_path = labels_path
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.num_gradient_steps = args.num_gradient_steps

        # DEEP LEARNING
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.sequence_length = args.sequence_length
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                print("Layer is Batchnorm!")
                self.has_BatchNorm = True
                break

        # Why do they use kwargs... that's annoying. I don't wanna pass it in every time...
        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        
        # My additional parameters
        ## Params that get logged
        self.pca_channels = args.pca_channels
        self.device_channels = args.device_channels
        self.lambdaF = args.lambdaF
        self.lambdaD = args.lambdaD
        self.lambdaE = args.lambdaE
        self.normalize_data = args.normalize_data
        self.local_round_threshold = args.local_round_threshold
        self.update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        self.final_idx = self.update_ix[-2] # eg drop last update from consideration, stop before it
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.smoothbatch_boolean = args.smoothbatch_boolean
        self.smoothbatch_learningrate = args.smoothbatch_learningrate
        ## Not logged params
        self.starting_update = args.starting_update
        self.current_update = args.starting_update # This is logged by a different var in the server
        self.dt = args.dt
        self.local_round = 0
        self.last_global_round = 0

        # Testing
        self.test_split_each_update = args.test_split_each_update # Not used...
        self.test_subj_IDs = args.test_subj_IDs # Uhhh is this even used rn...
        self.test_split_fraction = args.test_split_fraction # Not used...
        self.use_kfold_crossval = args.use_kfold_crossval
        self.num_kfolds = args.num_kfold_splits # Does clientbase need this...

        self.condition_number_lst = args.condition_number_lst
        if condition_number!=None:
            self.condition_number = condition_number
        elif len(self.condition_number_lst)==1:
            self.condition_number = self.condition_number_lst[0]
        else:
            self.condition_number = None
        self.verbose = args.verbose
        self.return_cost_func_comps = args.return_cost_func_comps
        self.run_train_metrics = args.run_train_metrics
        self.ndp = args.num_decimal_points
        
        self.loss_func = CPHSLoss(lambdaF=self.lambdaF, lambdaD=self.lambdaD, lambdaE=self.lambdaE)
        # This is the training loss log written to during CPHS_TRAINING_SUBROUTINE
        self.loss_log = []
        # Testing loss log written to directly within client.test_metrics()
        self.client_testing_log = []
        self.cost_func_comps_log = []
        self.gradient_norm_log = []
        #self.running_epoch_loss = []
        self.testing_clients = [] #??? Why does clientbase have this...

        if args.optimizer_str.upper() == "ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer_str.upper() == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer_str.upper() == "ADAGRAD":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer_str.upper() == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer_str.upper() == "ADAMW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            # Trying Adam as the default...
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.ewc_bool = args.ewc_bool
        self.fisher_mult = args.fisher_mult
        self.regularizers = None  # This gets set later, so this is fine for now...

        # Note this double dipping
        self.debug_mode = args.debug_mode
        self.check_loss_for_nan_inf = args.debug_mode


    # GOAL IS TO CONSOLIDATE EVERYTHING SHARED BETWEEN TRAIN_METRICS, TEST_METRICS, AND CPHS_TRAINING_SUBROUTINE
    def shared_loss_calc(self, x, y, model, train_mode=True):
        '''This simulates the data stream and then calculates the loss. DOES NOT BACKPROP'''
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)

        # Maybe this is causing problems for some reason in testing? Idk
        ## Maybe try replacing self.F with a toggle for self.F vs self.F_testing...
        ## Perhaps they share the same computational graph? ...
        F, V, y_ref = self.simulate_data_streaming_xy(x, y, input_model=model, train_mode=train_mode)

        # D@s = predicted velocity
        vel_pred = model(F)

        # L2 regularization term (from CPHS formulation)
        l2_loss = sum(torch.norm(param, p=2)**2 for name, param in model.named_parameters() if 'weight' in name)
        t1 = self.loss_func(vel_pred, y_ref)
        #if type(t1)==torch.Tensor:
        #    t1 = t1.sum()
        t2 = self.lambdaD*(l2_loss)
        t3 = self.lambdaF*(torch.linalg.matrix_norm((F))**2)
        #if type(t3)==torch.Tensor:
        #    t3 = t3.sum()

        if self.verbose or self.debug_mode:
            print(f"CB shared_loss_calc loss t1: {t1}")
            print(f"CB shared_loss_calc l2_loss: {l2_loss}")
        if self.check_loss_for_nan_inf:
            # It's working right now so I'll turn this off for the slight speed boost
            if np.isnan(t1.item()) or np.isinf(t1.item()):
                raise ValueError("CLIENTBASE: Error term is NAN/inf..")
            if np.isnan(t2.item()) or np.isinf(t2.item()):
                raise ValueError("CLIENTBASE: Decoder Effort term is NAN/inf...")
            if np.isnan(t3.item()) or np.isinf(t3.item()):
                raise ValueError("CLIENTBASE: User Effort term is NAN/inf...")

        # Apply additional regularizers (eg for continual learning)
        reg_sum = 0
        if self.regularizers!=None:
            print("Regularizers is not None: Applying additional regularizers!")
            if type(self.regularizers)==list:
                for reg_term in self.regularizers:
                    # This isn't quite correct, would need to be reg_term.penalty or something...
                    reg_sum += reg_term.penalty()
            else:
                reg_sum = self.regularizers.penalty()
        if self.verbose or self.debug_mode:
            print(f"CB shared_loss_calc reg_sum: {reg_sum}")

        # Only do this during training NOT TESTING EVAL!
        if train_mode:
            self.cost_func_comps_log = [(t1.item(), t2.item(), t3.item())]

        loss = t1 + t2 + t3 + reg_sum
        num_samples = x.size()[0]

        #print(f"t1: {t1.item():.6f}, t2: {t2.item():.6f}, t3: {t3.item()}, loss: {loss:.6f}, num_samples: {num_samples}\n")
        return loss, num_samples
    

    def cphs_training_subroutine(self, x, y):
        # Assert that the dataloader data corresponds to the correct update data
        # I think trainloader is fine so I can turn it off once tl has been verified
        #self.assert_tl_samples_match_npy(x, y)

        # Idk if train/test metrics need to be running this for APFL too
        ## PFL-NONIID did not...
        if self.algorithm!='APFL':
            # reset gradient so it doesn't accumulate
            self.optimizer.zero_grad()
        
        #print(f"CPHS_TRAINING_SUBROUTINE, USER {self.ID}")
        self.model.train()
        loss_obj, num_samples = self.shared_loss_calc(x, y, self.model, train_mode=True)

        if self.algorithm=='APFL':
            self.optimizer.zero_grad()
        
        # backward pass
        loss_obj.backward()
        self.loss_log.append(loss_obj.item()/num_samples)
        # This would need to be changed if you switch to a multi-layer model
        # Gradient norm
        if self.model_str == 'LinearRegression':
            weight_grad = self.model.weight.grad
            if weight_grad == None:
                print("Weight gradient is None...")
                grad_norm = -1
                self.gradient_norm_log.append(grad_norm)
            else:
                grad_norm = torch.linalg.norm(self.model.weight.grad, ord='fro') 
                self.gradient_norm_log.append(grad_norm)
        else:
            print(f"{self.model_str} is not LinearRegression: running separate grad norm extraction")
            grad_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            self.gradient_norm_log.append(grad_norm)
        #print(f"grad_norm: {grad_norm}")

        # update weights
        self.optimizer.step()

        if self.verbose:
            print(f"Client {self.ID}; update {self.current_update}; x.size(): {x.size()}; loss: {loss_obj.item():0.5f}")

    
    def simulate_data_streaming_xy(self, x, y, input_model, train_mode=True):
        '''
        Input:
            x, y --> A single training example from the trainloader

        Output: 
            Returns F, V, y_ref

        This function sets F (transformed input data) and V (Vplus used in cost func)
        Specifically: the loss function is its own class/object so it doesn't have 
            access to these (F and V)
        Must set F and V for model.output --> where? why? --> F is in the cost func, but where is V used? grad eval or something?
        '''

        s_temp = x
        p_reference = y

        # First, normalize the entire input data
        if self.normalize_data:
            # This is really scaling not norming
            s_normed = normalize_tensor(s_temp)
            p_reference = normalize_tensor(p_reference)
        else:
            s_normed = s_temp
        # Apply PCA if applicable
        if self.pca_channels!=self.device_channels:
            pca = PCA(n_components=self.pca_channels)
            s = torch.tensor(pca.fit_transform(s_normed), dtype=torch.float32)
        else:
            s = s_normed

        F = s[:-1,:]
        if train_mode:
            v_actual = input_model(s)
        else:
            # For testing, turn off the gradient tracking in the forward pass!
            with torch.no_grad():
                v_actual = input_model(s)
        # Numerical integration of v_actual to get p_actual
        p_actual = torch.cumsum(v_actual, dim=1)*self.dt
        V = (p_reference - p_actual)*self.dt # V doesn't actually get used...
        y_ref = p_reference[:-1, :]  # To match the input
        #^ This is used in t1 = self.loss_func(vel_pred, self.y_ref) in shared_loss_calc
        return F, V, y_ref


    def _load_train_data(self):
        '''
        This function actually loads the numpy data in, giving client access to its data (self variable). 
        This should only be run once on startup (or when round=0).
        Also does train/test split (default is holding out the last few updates)
        DOES NOT set/make a train-/data-loader
        '''

        if self.verbose:
            print(f"Client {self.ID} loading data file in [SHOULD ONLY RUN ONCE PER CLIENT]")
        # Load in client's data
        with open(self.samples_path, 'rb') as handle:
            samples_npy = np.load(handle)
        with open(self.labels_path, 'rb') as handle:
            labels_npy = np.load(handle)
        # Select for given condition #THIS IS THE ACTUAL TRAINING DATA AND LABELS FOR THE GIVEN TRIAL
        # THIS SHOULD ONLY APPLY TO SIMULATIONS NOT REAL TIME RUNS!
        starting_update_idx = self.update_ix[self.starting_update]
        self.cond_samples_npy = samples_npy[self.condition_number, starting_update_idx:self.final_idx, :]
        self.cond_labels_npy = labels_npy[self.condition_number, starting_update_idx:self.final_idx, :]
        # Split data into train and test sets
        if self.use_kfold_crossval:
            # Just set the training data to the full client dataset. KFold happens outside this code...
            ## It's gotta be outside the client so def not here

            # Set the number of examples (used to be done on init) --> ... THIS IS ABOUT TRAIN/TEST SPLIT
            ## These should really be renamed num_train_samples and num_test_samples
            self.train_samples = self.cond_samples_npy.shape[0]
            self.test_samples = None  # TODO: Shouldnt even be computed here...
            # Find the closest update idx (from predefined streamed batches in update_ix) to split the data at
            #self.test_split_idx = min(self.update_ix, key=lambda x:abs(x-testsplit_upper_bound))
            self.max_training_update_upbound = self.update_ix.index(self.update_ix[-1])  # Can I just remove this entirely... is used below...
        elif self.test_split_each_update:
            # Idk this might actually be supported just in a different function. I'm not sure. Don't plan on using it rn so who cares
            raise ValueError("test_split_each_update not supported yet")
        else: 
            # PREVIOUSLY, BY DEFAULT I witheld the last self.test_split_fraction% of EVERY CLIENT as the test set (for that client or for all clients...)
            # I think this really ought to be named traintestsplit_upperbound...
            testsplit_upper_bound = round((1-self.test_split_fraction)*(self.cond_samples_npy.shape[0]))
            # Set the number of examples (used to be done on init) --> ... THIS IS ABOUT TRAIN/TEST SPLIT
            self.train_samples = testsplit_upper_bound
            self.test_samples = self.cond_samples_npy.shape[0] - testsplit_upper_bound
            # Find the closest update idx (from predefined streamed batches in update_ix) to split the data at
            self.test_split_idx = min(self.update_ix, key=lambda x:abs(x-testsplit_upper_bound))
            self.max_training_update_upbound = self.update_ix.index(self.test_split_idx)
        

    def load_train_data(self, batch_size=None, eval=False, client_init=False):
        # Load full client dataasets
        if client_init:
            self._load_train_data()   # Returns nothing, sets self variables
        
        # Do I really want this here...
        if eval==False:
            # TODO: THIS SHOULD NOT BE UPDATED HERE LOL
            self.local_round += 1

            # Check if you need to advance the update
            # ---> THIS IMPLIES THAT I AM CREATING A NEW TRAINING LOADER FOR EACH UPDATE... this is what I want actually I think
            if (self.local_round%self.local_round_threshold==0) and (self.local_round>1) and (self.current_update < self.max_training_update_upbound):
                self.current_update += 1
                print(f"Client {self.ID} advances to update {self.current_update} on local round {self.local_round}")
            # Slice the full client dataset based on the current update number
            if self.current_update < self.max_training_update_upbound:
                # NOTE: Added the -self.update_ix[self.starting_update]
                self.update_lower_bound = self.update_ix[self.current_update] - self.update_ix[self.starting_update]
                self.update_upper_bound = self.update_ix[self.current_update+1] - self.update_ix[self.starting_update]
            else:
                self.update_lower_bound = self.update_ix[self.max_training_update_upbound - 1] - self.update_ix[self.starting_update]
                self.update_upper_bound = self.update_ix[self.max_training_update_upbound] - self.update_ix[self.starting_update]
        else:
            # There is no update, so no need to update/set the self.bounds above
            pass

        # Set the Dataset Obj
        # Creates a new TL each time, but doesn't have to re-read in the data. May not be optimal
        training_samples = self.cond_samples_npy[self.update_lower_bound:self.update_upper_bound,:]
        training_labels = self.cond_labels_npy[self.update_lower_bound:self.update_upper_bound,:]

        if self.deep_bool:
            if self.sequence_length*self.batch_size > training_samples.shape[0]:
                raise ValueError("seq_len*batch_size > num training samples: thus trainloader will be empty")
            training_dataset_obj = DeepSeqLenDataset(training_samples, training_labels, self.sequence_length)
        else:
            if self.batch_size > training_samples.shape[0]:
                raise ValueError("bs > num training samples: thus trainloader will be empty")
            training_dataset_obj = CustomEMGDataset(training_samples, training_labels)
        
        if self.verbose:
            print(f"cb load_train_data(): Client {self.ID}: Setting Training DataLoader")
        # Set dataloader
        if batch_size == None:
            batch_size = self.batch_size
        dl = DataLoader(
            dataset=training_dataset_obj,
            batch_size=batch_size, 
            drop_last=True,  # Yah idk if this should be true or false or if it matters...
            shuffle=False) 
        return dl


    def load_test_data(self, batch_size=None): 
        # Make sure this runs AFTER load_train_data so the data is already loaded in
        if self.verbose:
            print(f"Client {self.ID}: Setting Test DataLoader")
        if batch_size == None:
            batch_size = self.batch_size

        if self.use_kfold_crossval:
            # If client is in the val split and loads its testing data, all the data is testing
            # cond_samples_npy already starts at starting_update and ends at final_idx
            # TODO: Should I use a deepcopy or something so it's a different object... will this cause problems with the underlying computational graph...
            testing_samples = self.cond_samples_npy
            testing_labels = self.cond_labels_npy
            # How is mine different from UserTimeSeriesDataset?
            ## ^ Batching is built in by default for this one ... not sure how batching was done with the other one then...
            testing_dataset_obj = UserTimeSeriesDataset(testing_samples, testing_labels)#, batch_size=self.batch_size)
            return testing_dataset_obj
        elif self.test_split_each_update:
            testing_samples = self.cond_samples_npy[self.test_split_idx,:]
            testing_labels = self.cond_labels_npy[self.test_split_idx,:]

        if self.deep_bool:
            testing_dataset_obj = DeepSeqLenDataset(testing_samples, testing_labels, self.sequence_length)
        else:
            testing_dataset_obj = CustomEMGDataset(testing_samples, testing_labels)

        dl = DataLoader(
            dataset=testing_dataset_obj,
            batch_size=batch_size, 
            drop_last=True,  # Yah idk if this should be true or false or if it matters...
            shuffle=False) 
        return dl


    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()


    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()


    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test_metrics(self, saved_model_path=None, model_obj=None):
        '''Kai's docs: This function is for evaluating the model (on the testing data) during training
        Note that model.eval() is called so params aren't updated.
        
        Inputs:
            saved_model_path: full path (absolute or relative from PFL(\CB?)) to .pt model object
            OR
            model_obj:
            NOTE: setting both input params is unnecessary, only specify one. Otherwise an assertion will be raised
            ^ This is probably not ideal behaviour...

        NOTE: Should explicitly add a toggle for repeat testing (on the same dataset) using the global model or whatever, so it doesn't re-do val testing...
        '''

        if model_obj != None:
            eval_model = model_obj
        elif saved_model_path != None:
            eval_model = self.load_model(saved_model_path)
        else:
            # USING THE CLIENT'S LOCAL(/PERS) MODEL, NOT THE GLOBAL MODEL!
            ## This should be the default case, right?
            eval_model = self.model
        eval_model.to(self.device)
        eval_model.eval()

        total_loss = 0
        total_samples = 0
        if self.verbose:
            print(f'cb Client {self.ID} test_metrics()')
        with torch.no_grad():
            for i, (x, y) in enumerate(self.testloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                loss, num_samples = self.shared_loss_calc(x, y, eval_model, train_mode=False)
                # I think an issue could be if after streaming the values are getting "double-dipped"
                ## F, y_ref are geting used for testing as well, or rather the computational graph (gradient/history) remains and is what is getting double-dipped...
                print(f"(loss, num_samples): ({loss}, {num_samples})")

                total_loss += loss.item() * num_samples
                total_samples += num_samples
            print(f"TOTAL (total_loss, total_samples): ({total_loss}, {total_samples})")
            average_loss = total_loss / total_samples
            self.client_testing_log.append(average_loss)
            print(f"CALC'd AVERAGE: {average_loss}\n")
        return total_loss, total_samples # Return average_loss or total_loss...
    

    def train_metrics(self, saved_model_path=None, model_obj=None):
        '''Kai's docs: This function is for evaluating the model (on the training data for some reason) during training
        Note that model.eval() is called so params aren't updated.
        
        7/26/24: Why does this exist, just use the loss during actual training??? This is how PFL broke it up tho...'''
        if self.verbose:
            print("Client train_metrics()")

        if model_obj != None:
            eval_model = model_obj
        elif saved_model_path != None:
            eval_model = self.load_model(saved_model_path)
        else:
            eval_model = self.model
        eval_model.to(self.device)
        eval_model.eval()

        # TODO Fix this inefficiency!
        # ITS GOTTA BE SUPER INEFFICIENT TO RECREATE A NEW TL FOR EACH CLIENT EACH ROUND FOR TRAIN EVAL...
        ## How do I just reuse the existing training loader from the streamed training data?
        trainloader = self.load_train_data(eval=True)
        assert(len(trainloader)!=0)

        train_num = 0
        losses = 0
        if self.verbose:
            print(f'cb Client {self.ID} train_metrics()')
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                print(f'cb Client {self.ID} train_metrics() batch {i}')
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                #print(f"TRAIN_METRICS, USER {self.ID}")
                loss, num_samples = self.shared_loss_calc(x, y, eval_model)
                print(f"(loss, num_samples): ({loss}, {num_samples})")
                train_num += num_samples
                losses += loss.item() * num_samples
        print(f"TOTAL (losses, train_num): ({losses}, {train_num})\n")
        return losses, train_num

    
    def assert_tl_samples_match_npy(self, x, y, batch_num=None):  # I think batch_num should always work? As long as the batch size works out...
        # Assert that the dataloader data corresponds to the correct update data from the npy file that is loaded in
        # Right now this doesn't check the labels y... does print them tho

        nondl_x = np.round(self.cond_samples_npy[self.update_lower_bound:self.update_upper_bound], 4)
        nondl_y = np.round(self.cond_labels_npy[self.update_lower_bound:self.update_upper_bound], 4)
        if batch_num!=None:
            nondl_x = nondl_x[self.batch_size*batch_num:self.batch_size*batch_num+self.batch_size]
            nondl_y = nondl_y[self.batch_size*batch_num:self.batch_size*batch_num+self.batch_size]
        if (sum(sum(x[:5]-nondl_x[:5]))>0.01):  # 0.01 randomly chosen arbitarily small threshold
            # I think this bug is fixed now

            # ^Client11 fails when threshold is < 0.002, idk why there is any discrepancy
            # ^All numbers are positive so anything <1 is just rounding as far as I'm concerned
            print(f"clientavg: TRAINLOADER DOESN'T MATCH EXPECTED!! (@ update {self.current_update}, with x.size={x.size()})")
            print(f"Summed difference: {sum(sum(x[:5]-nondl_x[:5]))}")
            print(f"Trainloader x first 10 entries of channel 0: {x[:10, 0]}") 
            print(f"cond_samples_npy first 10 entries of channel 0: {nondl_x[:10, 0]}") 
            print()
            print(f"Trainloader y first 10 entries of channel 0: {y[:10, 0]}") 
            print(f"cond_labels_npy first 10 entries of channel 0: {nondl_y[:10, 0]}") 
            raise ValueError("Trainloader may not be working as anticipated")
        #assert(sum(sum(x[:5]-self.cond_labels_npy[self.update_ix[self.current_update]:self.update_ix[self.current_update+1]][:5]))==0) 


    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        # This just saves the model right...
        torch.save(item, os.path.join(item_path, self.ID + "_" + item_name + ".pt"))

        
    def load_item(self, item_name, item_path=None, full_path_to_item=None):
        if full_path_to_item!=None:
            # Uses torch.load so it assumes it is a model
            return torch.load(full_path_to_item)
        elif item_path != None:
            # Uses torch.load so it assumes it is a model...
            return torch.load(os.path.join(item_path, item_name))
            # An earlier default: 
            #return torch.load(os.path.join(item_path, self.ID + "_" + item_name + ".pt"))
        elif item_path == None:
            raise ValueError("No path (item_path or full_path_to_item) provided")
        

    def log_personalized_model_loss(self):
        '''Test the client's personalized, local model (self.model) after each training round and log it to a pers_loss_log'''
        pass
    