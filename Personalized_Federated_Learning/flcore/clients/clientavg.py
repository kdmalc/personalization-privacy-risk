# PFLNIID

import torch
#import torch.nn as nn
#import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.pflniid_utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, ID, samples_path, labels_path, condition_number, **kwargs):
        super().__init__(args, ID, samples_path, labels_path, condition_number, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        #self.model.train()

        # differential privacy
        #if self.privacy:
        #    self.model, self.optimizer, trainloader, privacy_engine = \
        #        initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()
        #if self.train_slow:
        #    max_local_steps = np.random.randint(1, max_local_steps // 2)

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

        #if self.privacy:
        #    eps, DELTA = get_dp_params(privacy_engine)
        #    print(f"Client {self.ID}", f"epsilon = {eps:.2f}, sigma = {DELTA}")