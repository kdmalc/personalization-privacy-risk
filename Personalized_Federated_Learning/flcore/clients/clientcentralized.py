# PFLNIID

import torch
import time
from flcore.clients.clientbase import Client
#import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from utils.processing_funcs import normalize_tensor
from utils.emg_dataset_class import *


class clientCent(Client):
    def __init__(self, args, ID, dir_path, condition_number_lst):
        super().__init__(args, ID, None, None, condition_number_lst)

        self.dir_path = dir_path

        # training_dataset_obj = CustomEMGDataset(self.cond_samples_npy[self.update_lower_bound:self.update_upper_bound,:], self.cond_labels_npy[self.update_lower_bound:self.update_upper_bound,:])
        # Need to deal with these variables... not doing streaming right now...
        #self.cond_samples_npy
        #self.cond_labels_npy
        #self.update_lower_bound = 0
        #self.update_upper_bound = -1


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
        # Get a list of all .npy files in the folder
        #S107_TrainData_8by20770by64
        #S107_Labels_8by20770by2
        file_list = [f for f in os.listdir(self.dir_path) if f.endswith('.npy')]
        # Initialize an empty array to store the concatenated data
        giant_array = np.empty((0,))
        # Iterate through each .npy file and horizontally concatenate to the giant array
        for file_name in file_list:
            file_path = os.path.join(self.dir_path, file_name)
            data_array = np.load(file_path)
            if "Labels" in file_name:
                self.cond_labels_npy = data_array[self.condition_number,:,:]
            elif "Data" in file_name:
                self.cond_samples_npy = data_array[self.condition_number,:,:]
            else:
                raise ValueError("Did not find Labels or Data in filename...")
            
            giant_array = np.hstack((giant_array, data_array))

        # Split data into train and test sets
        if self.test_split_users:
            # NOT FINISHED YET
            raise ValueError("test_split_users not supported yet.  Stat hetero concern")
            # Randomly pick the test_split_fraction of users to be completely held out of training to be used for testing
            num_test_users = round(len(self.clients)*self.test_split_fraction)
            # Pick/sample num_test_users from self.clients to be removed and put into self.testing_clients
            self.testing_clients = [self.clients.pop(random.randrange(len(self.clients))) for _ in range(num_test_users)]
            # ^Hmmm this requires a full rewrite of the code... 
        elif self.test_split_each_update:
            # Idk this might actually be supported just in a different function. I'm not sure. Don't plan on using it rn so who cares
            raise ValueError("test_split_each_update not supported yet.  Idk if this is necessary to add")
        else: 
            testsplit_upper_bound = round((1-self.test_split_fraction)*(self.cond_samples_npy.shape[0]))
        # Set the number of examples (used to be done on init) --> ... THIS IS ABOUT TRAIN/TEST SPLIT
        self.train_samples = testsplit_upper_bound
        self.test_samples = self.cond_samples_npy.shape[0] - testsplit_upper_bound
        train_test_update_number_split = min(self.update_ix, key=lambda x:abs(x-testsplit_upper_bound))
        self.max_training_update_upbound = self.update_ix.index(train_test_update_number_split)
        self.test_split_idx = self.update_ix[self.max_training_update_upbound]
        

    # THIS IS ONLY CALLED IN TRAIN_METRICS()...
    def load_train_data(self, batch_size=None, eval=False, client_init=False):
        # Load full client dataasets
        if client_init:
            self._load_train_data()   # Returns nothing, sets self variables
        '''
        # Do I really want this here...
        if eval==False:
            self.local_round += 1
            # Check if you need to advance the update
            # ---> THIS IMPLIES THAT I AM CREATING A NEW TRAINING LOADER FOR EACH UPDATE... this is what I want actually I think
            if (self.local_round%self.local_round_threshold==0) and (self.local_round>1) and (self.current_update < self.max_training_update_upbound):
                self.current_update += 1
                print(f"Client {self.ID} advances to update {self.current_update} on local round {self.local_round}")
            # Slice the full client dataset based on the current update number
            if self.current_update < self.max_training_update_upbound:
                self.update_lower_bound = self.update_ix[self.current_update]
                self.update_upper_bound = self.update_ix[self.current_update+1]
            else:
                self.update_lower_bound = self.update_ix[self.max_training_update_upbound - 1]
                self.update_upper_bound = self.update_ix[self.max_training_update_upbound]
        '''

        # Set the Dataset Obj
        # Creates a new TL each time, but doesn't have to re-read in the data. May not be optimal
        training_dataset_obj = CustomEMGDataset(self.cond_samples_npy[self.update_lower_bound:self.update_upper_bound,:], self.cond_labels_npy[self.update_lower_bound:self.update_upper_bound,:])
        X_data = torch.tensor(training_dataset_obj['x'], dtype=torch.float32)
        y_data = torch.tensor(training_dataset_obj['y'], dtype=torch.float32)
        training_data_for_dataloader = [(x, y) for x, y in zip(X_data, y_data)]
        
        if self.verbose:
            print(f"cb load_train_data(): Client {self.ID}: Setting Training DataLoader")
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
        if self.verbose:
            print(f"Client {self.ID}: Setting Test DataLoader")
        if batch_size == None:
            batch_size = self.batch_size

        #test_data = read_client_data(self.dataset, self.ID, self.current_update, is_train=False)
        testing_dataset_obj = CustomEMGDataset(self.cond_samples_npy[self.test_split_idx:,:], self.cond_labels_npy[self.test_split_idx:,:])
        X_data = torch.Tensor(testing_dataset_obj['x']).type(torch.float32)
        y_data = torch.Tensor(testing_dataset_obj['y']).type(torch.float32)
        testing_data_for_dataloader = [(x, y) for x, y in zip(X_data, y_data)]

        dl = DataLoader(
            dataset=testing_data_for_dataloader,
            batch_size=batch_size, 
            drop_last=False,  # Yah idk if this should be true or false or if it matters...
            shuffle=False) 
        return dl


    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        #self.model.train()
        
        start_time = time.time()

        if self.verbose:
            print(f'Client {self.ID} Training')
        # Save the client's starting weights for Smoothbatch
        if self.smoothbatch_boolean:
            starting_weights = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    starting_weights[name] = param.data.clone()
        for epoch in range(self.local_epochs):
            for step in range(self.num_gradient_steps):
                # Currently, each tl only has 1 batch of 1200 [eg 1 update] (8/5/23)
                # ^ I don't think this is relevant anymore...?
                for i, (x, y) in enumerate(trainloader):
                    if self.verbose:
                        print(f"Epoch {epoch}, grad step {step}, batch {i}")
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
