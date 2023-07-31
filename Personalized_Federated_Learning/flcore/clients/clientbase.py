# PFLNIID

import copy
import torch
import torch.nn as nn
import numpy as np
import random
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from sklearn.decomposition import PCA

from flcore.pflniid_utils.data_utils import read_client_data
from utils.custom_loss_class import CPHSLoss
from utils.emg_dataset_class import *


#https://www.youtube.com/watch?v=3GVUzwXXihs
#^ Very helpful video about samplers and getting data from DataLoaders
# We will just stick with sequential, which he doesn't show, but it is obvious how it works after seeing the vid


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, ID, samples_path, labels_path, **kwargs):
        self.model = copy.deepcopy(args.model)
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
        self.pca_channels = args.pca_channels
        self.device_channels = args.device_channels
        self.lambdaF = args.lambdaF
        self.lambdaD = args.lambdaD
        self.lambdaE = args.lambdaE
        self.current_update = args.starting_update
        self.dt = args.dt
        self.normalize_emg = args.normalize_emg
        self.normalize_V = args.normalize_V
        self.local_round = 0
        self.last_global_round = 0
        self.local_round_threshold = args.local_round_threshold
        self.update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        self.test_split_fraction = args.test_split_fraction
        self.test_split_each_update = args.test_split_each_update
        self.test_split_users = args.test_split_users
        assert (not (self.test_split_users and self.test_split_each_update)), "test_split_users and test_split_each_update cannot both be true (contradictory test conditions)"
        self.condition_number = args.condition_number
        self.verbose = args.verbose
        self.return_cost_func_comps = args.return_cost_func_comps

        # Before this I need to run the INIT update segmentation code...
        init_dl = self.load_train_data()
        self.simulate_data_streaming(init_dl)
        # ^ This func sets F, V, etc
        
        self.loss = CPHSLoss(self.F, self.model.weight, self.V, self.F.size()[1], lambdaF=self.lambdaF, lambdaD=self.lambdaD, lambdaE=self.lambdaE, Nd=2, Ne=self.pca_channels, return_cost_func_comps=self.return_cost_func_comps)
        self.loss_log = []
        self.cost_func_comps_log = []
        #self.running_epoch_loss = []
        self.testing_clients = []

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        
        
    def simulate_data_streaming(self, dl, test_data=False):
        it = iter(dl)
        s0 = it.__next__()
        # self.max_training_update_upbound
        # I think this is wrong, implies it never advances the update? Didnt I handle that somewhere else tho...
        s_temp = s0[0][0:self.update_ix[1],:]
        p_reference = torch.transpose(s0[1][0:self.update_ix[1],:], 0, 1)

        # First, normalize the entire s matrix
        if self.normalize_emg:
            s_normed = s_temp / torch.linalg.norm(s_temp, ord='fro')
            assert (torch.linalg.norm(s_normed, ord='fro')<1.2) and (torch.linalg.norm(s_normed, ord='fro')>0.8)
        else:
            s_normed = s_temp
        # Apply PCA if applicable
        if self.pca_channels!=self.device_channels:  # 64 is the number of channels present on the recording armband
            pca = PCA(n_components=self.pca_channels)
            s = torch.transpose(torch.tensor(pca.fit_transform(s_normed), dtype=torch.float32), 0, 1)
        else:
            s = torch.transpose(s_normed, 0, 1)

        self.F = s[:,:-1]
        v_actual =  torch.matmul(self.model.weight, s)
        p_actual = torch.cumsum(v_actual, dim=1)*self.dt  # Numerical integration of v_actual to get p_actual
        self.V = (p_reference - p_actual)*self.dt
        if self.normalize_V:
            self.V = self.V/torch.linalg.norm(self.V, ord='fro')
            assert (torch.linalg.norm(self.V, ord='fro')<1.2) and (torch.linalg.norm(self.V, ord='fro')>0.8)
        self.Y = p_reference[:, :-1]  # To match the input


    def _load_train_data(self):
        if self.verbose:
            print(f"Client{self.ID} loading data file in [SHOULD ONLY RUN ONCE PER CLIENT]")
        # Load in client's data
        with open(self.samples_path, 'rb') as handle:
            samples_npy = np.load(handle)
        with open(self.labels_path, 'rb') as handle:
            labels_npy = np.load(handle)
        # Select for given condition #THIS IS THE ACTUAL TRAINING DATA AND LABELS FOR THE GIVEN TRIAL
        self.cond_samples_npy = samples_npy[self.condition_number,:,:]
        self.cond_labels_npy = labels_npy[self.condition_number,:,:]
        # Split data into train and test sets
        if self.test_split_users:
            # NOT FINISHED YET
            # Randomly pick the test_split_fraction of users to be completely held out of training to be used for testing
            num_test_users = round(len(self.clients)*self.test_split_fraction)
            # Pick/sample num_test_users from self.clients to be removed and put into self.testing_clients
            self.testing_clients = [self.clients.pop(random.randrange(len(self.clients))) for _ in range(num_test_users)]
            # Hmmm this requires a full rewrite of the code... 
            raise ValueError("test_split_users is not fully supported yet")
        elif self.test_split_each_update:
            # Idk this might actually be supported just in a different function. I'm not sure. Don't plan on using it rn so who cares
            raise ValueError("test_split_each_update not supported yet.  Idk if this is necessary to add")
        else: 
            testsplit_upper_bound = round((1-self.test_split_fraction)*(self.cond_samples_npy.shape[0]))
        # Set the number of examples (used to be done on init) --> ... THIS IS ABOUT TRAIN/TEST SPLIT
        self.train_samples = testsplit_upper_bound
        self.test_samples = self.cond_samples_npy.shape[0] - testsplit_upper_bound
        # The below gets stuck in the debugger and just keeps running until you step over
        train_test_update_number_split = min(self.update_ix, key=lambda x:abs(x-testsplit_upper_bound))
        self.max_training_update_upbound = self.update_ix.index(train_test_update_number_split)
        

    def load_train_data(self, batch_size=None):
        # Load full client dataasets
        if self.local_round == 0:
            self._load_train_data()   # Returns nothing, sets self variables
            if self.current_update < self.max_training_update_upbound:
                self.update_lower_bound = self.update_ix[self.current_update]
                self.update_upper_bound = self.update_ix[self.current_update+1]

        self.local_round += 1
        # Check if you need to advance the update
        # ---> THIS IMPLIES THAT I AM CREATING A NEW TRAINING LOADER FOR EACH UPDATE...
        if (self.local_round>1) and (self.current_update < 16) and (self.local_round%self.local_round_threshold==0):
            self.current_update += 1
            print(f"Client{self.ID} advances to update {self.current_update}")
            # Slice the full client dataset based on the current update number
            if self.current_update < self.max_training_update_upbound:
                self.update_lower_bound = self.update_ix[self.current_update]
                self.update_upper_bound = self.update_ix[self.current_update+1]
            else:
                self.update_lower_bound = self.max_training_update_upbound - 1
                self.update_upper_bound = self.max_training_update_upbound
        # Set the Dataset Obj
        # Uhhhh is this creating a new one each time? As long as its not re-reading in the data it probably doesn't matter...
        #train_data = read_client_data(self.dataset, self.ID, self.current_update, is_train=True)  # Original code
        #CustomEMGDataset(emgs_block1[my_user][condition_idx,update_lower_bound:update_upper_bound,:], refs_block1[my_user][condition_idx,update_lower_bound:update_upper_bound,:])
        training_dataset_obj = CustomEMGDataset(self.cond_samples_npy[self.update_lower_bound:self.update_upper_bound,:], self.cond_labels_npy[self.update_lower_bound:self.update_upper_bound,:])
        X_data = torch.Tensor(training_dataset_obj['x']).type(torch.float32)
        y_data = torch.Tensor(training_dataset_obj['y']).type(torch.float32)
        training_data_for_dataloader = [(x, y) for x, y in zip(X_data, y_data)]
        
        if self.verbose:
            print(f"clientbase load_train_data(): Client{self.ID}: Setting Training DataLoader")
        # Set dataloader
        if batch_size == None:
            batch_size = self.batch_size
        dl = DataLoader(
            dataset=training_data_for_dataloader,
            batch_size=batch_size, 
            drop_last=False,  # Yah idk if this should be true or false or if it matters...
            shuffle=False) 
        return dl

    def load_test_data(self, batch_size=None): 
        # Make sure this runs AFTER load_train_data so the data is already loaded in
        print(f"Client{self.ID}: Setting Test DataLoader")
        if batch_size == None:
            batch_size = self.batch_size

        #test_data = read_client_data(self.dataset, self.ID, self.current_update, is_train=False)
        testing_dataset_obj = CustomEMGDataset(self.cond_samples_npy[self.update_upper_bound:,:], self.cond_labels_npy[self.update_upper_bound:,:])
        X_data = torch.Tensor(testing_dataset_obj['x']).type(torch.float32)
        y_data = torch.Tensor(testing_dataset_obj['y']).type(torch.float32)
        testing_data_for_dataloader = [(x, y) for x, y in zip(X_data, y_data)]

        dl = DataLoader(
            dataset=testing_data_for_dataloader,
            batch_size=batch_size, 
            drop_last=False,  # Yah idk if this should be true or false or if it matters...
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

    def test_metrics(self, saved_model=None):
        '''Kai's docs: This function is for evaluating the model (on the testing data) during training
        Note that model.eval() is called so params aren't updated.'''

        if saved_model != None:
            self.model = self.load_model(saved_model)
        self.model.to(self.device)

        testloaderfull = self.load_test_data()
        # Should I be simulate streaming with the testing data... 
        # no the defualt should be holding a subj or two out and testing on them...
        # Maybe it doesnt matter as much since I'm not doing classification, so bias wouldn't be subject level but rather time/task-progress level
        #self.simulate_data_streaming(testloaderfull)
        self.model.eval()

        running_test_loss = 0
        num_samples = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(testloaderfull):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_loss = self.loss(output, y, self.model)[0].item()  # Just get the actual loss function term
                running_test_loss += test_loss
                print(f"clientbase test_metrics() batch: {i}, loss: {test_loss:0,.1f}")
                num_samples += x.size()[0]
            
        return running_test_loss, num_samples
    

    def train_metrics(self):
        '''Kai's docs: This function is for evaluating the model (on the training data for some reason) during training
        Note that model.eval() is called so params aren't updated.'''

        if self.verbose:
            print("Client train_metrics()")
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        counter = 0
        with torch.no_grad():
            for x, y in trainloader:
                counter += 1
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y, self.model)
                if self.return_cost_func_comps:
                    loss = loss[0]
                #if self.verbose:
                #    pass
                print(f"Client{self.ID} train_metrics() batch: {counter}, loss: {loss:0,.1f}")
                train_num += y.shape[0]  # Why is this y.shape and not x.shape?... I guess they are the same row dims?
                # Why are they multiplying by y.shape[0] here...
                losses += loss.item() #* y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

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
        torch.save(item, os.path.join(item_path, "client_" + str(self.ID) + "_" + item_name + ".pt"))

        
    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.ID) + "_" + item_name + ".pt"))
    
    
    ##############################################################################
    def test_metrics_archive(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))