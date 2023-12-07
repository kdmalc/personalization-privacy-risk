# PFLNIID

import torch
import time
from flcore.clients.clientbase import Client
#import torch.nn as nn
import numpy as np
import os
#import random
from torch.utils.data import DataLoader
from utils.emg_dataset_class import *


class clientCent(Client):
    def __init__(self, args, ID, dir_path, condition_number_lst, **kwargs):
        super().__init__(args, ID, None, None, condition_number_lst, **kwargs)

        self.dir_path = dir_path
        self.model_str = args.model
        self.test_subj_IDs = args.test_subj_IDs
        self.test_sIDs = [fullID.split('_')[1] for fullID in self.test_subj_IDs]

        ############################################################################
        self.train_loader = self.load_train_data(client_init=True) # batch_size=None
        print("TRAINLOADER")
        print(f"self.cond_labels_npy.shape: {self.cond_labels_npy.shape}")
        print(f"self.cond_samples_npy.shape: {self.cond_samples_npy.shape}")
        for i, (x, y) in enumerate(self.train_loader):
            if i==0:
                print(f'Batch {i}, Batch Size: {x.shape}')

        self.test_loader = self.load_test_data()
        print("TESTLOADER")
        print("TESTLOADER Length:", len(self.test_loader))
        for i, (x, y) in enumerate(self.test_loader):
            if i==0:
                print(f'Batch {i}, Batch Size: {x.shape}')
        ############################################################################


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
        # Get a list of all .npy files in the folder (of form S107_TrainData_8by20770by64, S107_Labels_8by20770by2)
        file_list = [f for f in os.listdir(self.dir_path) if f.endswith('.npy')]
        # Initialize an empty array to store the concatenated data
        self.cond_labels_npy = np.empty((0, 0))
        self.cond_samples_npy = np.empty((0, 0))
        self.test_labels = np.empty((0, 0))
        self.test_samples = np.empty((0, 0))
        # Iterate through each .npy file and horizontally concatenate to the giant array
        for file_name in file_list:
            file_path = os.path.join(self.dir_path, file_name)
            data_array = np.load(file_path)
            for cond_num in self.condition_number_lst:
                # This is a bit of a hardcoded soln...
                # Extract the subject ID from the string
                subject_id = file_name.split('_')[0]
                if (self.test_split_users==True) and (subject_id in self.test_sIDs):
                    if "Labels" in file_name:
                        loaded_labels = data_array[cond_num-1,:,:]
                        if self.test_labels.size == 0:
                            self.test_labels = loaded_labels
                        else:
                            self.test_labels = np.vstack((self.test_labels, loaded_labels))
                    elif "Data" in file_name:
                        loaded_samples = data_array[cond_num-1,:,:]
                        if self.test_samples.size == 0:
                            self.test_samples = loaded_samples
                        else:
                            self.test_samples = np.vstack((self.test_samples, loaded_samples))
                    else:
                        raise ValueError("Did not find Labels or Data in filename...")
                else:    
                    if "Labels" in file_name:
                        loaded_labels = data_array[cond_num-1,:,:]
                        if self.cond_labels_npy.size == 0:
                            self.cond_labels_npy = loaded_labels
                        else:
                            self.cond_labels_npy = np.vstack((self.cond_labels_npy, loaded_labels))
                    elif "Data" in file_name:
                        loaded_samples = data_array[cond_num-1,:,:]
                        if self.cond_samples_npy.size == 0:
                            self.cond_samples_npy = loaded_samples
                        else:
                            self.cond_samples_npy = np.vstack((self.cond_samples_npy, loaded_samples))
                    else:
                        raise ValueError("Did not find Labels or Data in filename...")
        

    def load_train_data(self, batch_size=None, eval=False, client_init=False):
        # Load full client dataasets
        if client_init:
            self._load_train_data()   # Returns nothing, sets self variables

        # Set the Dataset Obj
        if self.model_str == "LinearRegression":
            training_dataset_obj = CustomEMGDataset(self.cond_samples_npy, self.cond_labels_npy)
            #X_data = torch.tensor(training_dataset_obj['x'], dtype=torch.float32)
            #y_data = torch.tensor(training_dataset_obj['y'], dtype=torch.float32)
            #training_data_for_dataloader = [(x, y) for x, y in zip(X_data, y_data)]
        else:
            training_dataset_obj = BasicDataset(self.cond_samples_npy, self.cond_labels_npy, self.sequence_length)

        #print("TRAINING Dataset Length:", len(training_dataset_obj))
        
        if self.verbose:
            print(f"cb load_train_data(): Client {self.ID}: Setting Training DataLoader")
        # Set dataloader
        if batch_size == None:
            batch_size = self.batch_size
        dl = DataLoader(
            dataset=training_dataset_obj,
            batch_size=batch_size, 
            drop_last=True,
            shuffle=False) 
        return dl

    def load_test_data(self, batch_size=None): 
        # Make sure this runs AFTER load_train_data so the data is already loaded in
        if batch_size == None:
            batch_size = self.batch_size

        #print(f"self.test_samples.shape: {self.test_samples.shape}")
        #print(f"self.test_labels.shape: {self.test_labels.shape}")

        if self.model_str == "LinearRegression":
            testing_dataset_obj = CustomEMGDataset(self.test_samples, self.test_labels)
        else:
            testing_dataset_obj = BasicDataset(self.test_samples, self.test_labels, self.sequence_length)

        #print("TESTING Dataset Length:", len(testing_dataset_obj))

        dl = DataLoader(
            dataset=testing_dataset_obj,
            batch_size=batch_size, 
            drop_last=True,
            shuffle=False) 
        return dl
    

    def test_metrics(self, saved_model_path=None, model_obj=None):
        '''Kai's docs: This function is for evaluating the model (on the testing data) during training
        Note that model.eval() is called so params aren't updated.
        
        Inputs:
            saved_model_path: full path (absolute or relative from PFL(\CB?)) to .pt model object
            OR
            model_obj:
            NOTE: setting both input params is unnecessary, only specify one. Otherwise an assertion will be raised
            '''

        if model_obj != None:
            eval_model = model_obj
        elif saved_model_path != None:
            eval_model = self.load_model(saved_model_path)
        else:
            eval_model = self.model
        eval_model.to(self.device)
        eval_model.eval()

        running_test_loss = 0
        num_samples = 0
        if self.verbose:
            print(f'cb Client {self.ID} test_metrics()')

        ###########################################################################################
        #print("CENTRALIZED TEST_METRICS!")
        ###########################################################################################

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.simulate_data_streaming_xy(x, y)
                # ^ Idk if this actaully needs to be called again here...
                # D@s = predicted velocity
                vel_pred = eval_model(self.F)

                # L2 regularization term
                l2_loss = 0
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        l2_loss += torch.norm(param, p=2)
                t1 = self.loss_func(vel_pred, self.y_ref)
                if type(t1)==torch.Tensor:
                    t1 = t1.sum()
                t2 = self.lambdaD*(l2_loss**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                if type(t3)==torch.Tensor:
                    t3 = t3.sum()
                loss = t1 + t2 + t3

                test_loss = loss.item()  # Just get the actual loss function term
                running_test_loss += test_loss
                #print(f"batch {i}, loss {test_loss:0,.5f}")
                num_samples += x.size()[0]
        self.client_testing_log.append(running_test_loss / num_samples)   
        return running_test_loss, num_samples
    

    def train_metrics(self, saved_model_path=None, model_obj=None):
        '''Kai's docs: This function is for evaluating the model (on the training data for some reason) during training
        Note that model.eval() is called so params aren't updated.'''
        if model_obj != None:
            eval_model = model_obj
        elif saved_model_path != None:
            eval_model = self.load_model(saved_model_path)
        else:
            eval_model = self.model
        eval_model.eval()

        train_num = 0
        losses = 0
        if self.verbose:
            print(f'cb Client {self.ID} train_metrics()')
        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.simulate_data_streaming_xy(x, y)
                vel_pred = eval_model(self.F)

                # L2 regularization term
                l2_loss = 0
                for name, param in eval_model.named_parameters():
                    if 'weight' in name:
                        l2_loss += torch.norm(param, p=2)
                t1 = self.loss_func(vel_pred, self.y_ref)
                if type(t1)==torch.Tensor:
                    t1 = t1.sum()
                t2 = self.lambdaD*(l2_loss**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                if type(t3)==torch.Tensor:
                    t3 = t3.sum()
                loss = t1 + t2 + t3
                if self.verbose:
                    print(f"batch {i}, loss {loss:0,.5f}")
                train_num += self.y_ref.shape[0]
                # ^ This is probably wrong now since I removed the transpose... (12/2/23)
                losses += loss.item() #* y.shape[0]
        return losses, train_num


    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        #self.model.train()
        
        start_time = time.time()

        # Save the client's starting weights for Smoothbatch
        if self.smoothbatch_boolean:
            starting_weights = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    starting_weights[name] = param.data.clone()

        for epoch in range(self.local_epochs):
            for step in range(self.num_gradient_steps):
                print(f"Epoch {epoch}, grad step {step}")
                for i, (x, y) in enumerate(trainloader):
                    #print(f"Batch {i}, x.shape: {x.shape}, y.shape: {y.shape}")
                    if x.numel() != 0:
                        self.cphs_training_subroutine(x, y)
        # Do SmoothBatch if applicable
        if self.smoothbatch_boolean:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data = self.smoothbatch_learningrate*starting_weights[name] + (1 - self.smoothbatch_learningrate)*param.data

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


