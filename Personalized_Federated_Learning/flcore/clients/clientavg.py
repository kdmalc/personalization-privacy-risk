# PFLNIID

import torch
#import torch.nn as nn
#import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.pflniid_utils.privacy import *

from sklearn.decomposition import PCA
from utils.processing_funcs import normalize_tensor


class clientAVG(Client):
    def __init__(self, args, ID, samples_path, labels_path, condition_number, **kwargs):
        super().__init__(args, ID, samples_path, labels_path, condition_number, **kwargs)

    def train(self):
        self.load_train_data()
        self.model.train()
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
                # Currently, each tl has 1200 (or 1202...) samples [eg 1 update] (1/13/24)
                for i, (x, y) in enumerate(self.trainloader):
                    if self.verbose:
                        print(f"Epoch {epoch}, grad step {step}, batch {i}")
                    #self.cphs_training_subroutine(x, y)
                    self.optimizer.zero_grad()
                    self.model.train() # Does this need to be here...
                    #loss_obj, num_samples = self.shared_loss_calc(x, y, self.model, train_mode=True)
                    #F, V, y_ref = self.simulate_data_streaming_xy(x, y, input_model=model, train_mode=train_mode)
                    s_temp = x
                    p_reference = y
                    if self.normalize_data:
                        s_normed = normalize_tensor(s_temp)
                        p_reference = normalize_tensor(p_reference)
                    else:
                        s_normed = s_temp
                    if self.pca_channels!=self.device_channels:
                        pca = PCA(n_components=self.pca_channels)
                        s = torch.tensor(pca.fit_transform(s_normed), dtype=torch.float32)
                    else:
                        s = s_normed
                    F = s[:-1,:]
                    v_actual = self.model(s)
                    p_actual = torch.cumsum(v_actual, dim=1)*self.dt
                    #V = (p_reference - p_actual)*self.dt
                    y_ref = p_reference[:-1, :]  # To match the input
                    #return F, V, y_ref    
                    vel_pred = self.model(F)
                    l2_loss = sum(torch.norm(param, p=2)**2 for name, param in self.model.named_parameters() if 'weight' in name)
                    t1 = self.loss_func(vel_pred, y_ref)
                    t2 = self.lambdaD*(l2_loss)
                    t3 = self.lambdaF*(torch.linalg.matrix_norm((F))**2)
                    self.cost_func_comps_log = [(t1.item(), t2.item(), t3.item())]
                    loss_obj = t1 + t2 + t3 #+ reg_sum
                    num_samples = x.size()[0]
                    #return loss, num_samples
                    loss_obj.backward()
                    self.loss_log.append(loss_obj.item()/num_samples)
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
                    self.optimizer.step()
                    if self.smoothbatch_boolean:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    param.data = self.smoothbatch_learningrate*starting_weights[name] + (1 - self.smoothbatch_learningrate)*param.data

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time