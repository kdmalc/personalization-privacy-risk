# PFLNIID

import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.pflniid_utils.data_utils import read_client_data
from utils.custom_loss_class import CPHSLoss


#https://www.youtube.com/watch?v=3GVUzwXXihs
#^ Very helpful video about samplers and getting data from DataLoaders
# We will just stick with sequential, which he doesn't show, but it is obvious how it works after seeing the vid


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, ID, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.ID = ID  # integer
        self.save_folder_name = args.save_folder_name

        #self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        # KAI: Idk what this is doing...
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        
        # My additional parameters
        self.pca_channels = args.pca_channels
        self.device_channels = args.device_channels
        self.lambdaF = args.lambdas[0]
        self.lambdaD = args.lambdas[1]
        self.lambdaE = args.lambdas[2]
        self.current_update = args.starting_update
        self.dt = args.dt
        self.normalize_emg = args.normalize_emg
        self.normalize_V = args.normalize_V
        self.local_round = 0
        self.last_global_round = 0
        self.local_round_threshold = args.local_round_threshold
        self.update_ix=[0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        
        # Before this I need to run the INIT update segmentation code...
        init_dl = self.load_train_data()
        # Before this I need to run the INIT update segmentation code...
        #train_data = read_client_data(dataset, ID, is_train=True)
        #dl = DataLoader(
        #dataset=train_data,
        #batch_size=batch_size, 
        #drop_last=False) 
        init_it = iter(init_dl)
        
        s0 = init_it.__next__()
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
            V = V/torch.linalg.norm(V, ord='fro')
            assert (torch.linalg.norm(V, ord='fro')<1.2) and (torch.linalg.norm(V, ord='fro')>0.8)
        y = p_reference[:, :-1]  # To match the input
        
        self.loss = CPHSLoss(self.F, self.model.weight, self.V, torch.view(self.F)[0], lambdaF=self.lambdaF, lambdaD=self.lambdaD, lambdaE=self.lambdaE, Nd=2, Ne=self.pca_channels, return_cost_func_comps=False)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        self.local_round += 1
        if (self.current_update < 17) and (self.local_round%self.local_round_threshold==0):
            self.current_update += 1
        
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.ID, is_train=True)
        dl = DataLoader(
            dataset=train_data,
            batch_size=batch_size, 
            drop_last=False,  # Yah idk if this should be true or false or if it matters...
            shuffle=False) 
        return dl

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.ID, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
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

    def test_metrics(self):
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

    def train_metrics(self):        
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

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

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))