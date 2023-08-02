# PFLNIID

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.pflniid_utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, ID, train_samples, test_samples, **kwargs):
        super().__init__(args, ID, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        #if self.privacy:
        #    self.model, self.optimizer, trainloader, privacy_engine = \
        #        initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_epochs
        #if self.train_slow:
        #    max_local_steps = np.random.randint(1, max_local_steps // 2)

        # WHICH OF THESE LOOPS IS EQUIVALENT TO MY EPOCHS...
        print(f'Client{self.ID} Training')
        running_num_samples = 0
        for step in range(max_local_steps):  # I'm assuming this is gradient steps?... are local epochs the same as gd steps?
            for i, (x, y) in enumerate(trainloader):  # This is all the data in a given batch, I think? Can I just kill this... PITA
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                #if self.train_slow:
                #    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                #print(f"clientAVG ----> Training LOSS {i}")  # What is this even tellimg me lol
                loss = self.loss(output, y, self.model)
                if self.return_cost_func_comps:
                    self.cost_func_comps_log.append(loss[1:])
                    loss = loss[0]
                else:
                    # .item() ONLY WORKS WITH 1D TENSORS!!!
                    t1 = self.loss.term1_error.item()
                    t2 = self.loss.term2_ld_decnorm.item()
                    t3 = self.loss.term3_lf_emgnorm.item()
                    if np.isnan(t1):
                        print("CLIENTAVG: Error term is None...")
                        t1 = -1
                    if np.isnan(t2):
                        print("CLIENTAVG: Decoder Effort term is None...")
                        t2 = -1
                    if np.isnan(t3):
                        print("CLIENTAVG: User Effort term is None...")
                        t3 = -1
                    self.cost_func_comps_log.append((t1, t2, t3))
                print(f"Step {step}, pair {i} in traindl; x.size(): {x.size()}; loss: {loss.item():0.2f}")
                self.loss_log.append(loss.item())
                #self.running_epoch_loss.append(loss.item() * x.size(0))  # From: running_epoch_loss.append(loss.item() * images.size(0))
                running_num_samples += x.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient norm
                weight_grad = self.model.weight.grad
                if weight_grad == None:
                    print("Weight gradient is None...")
                    self.gradient_norm_log.append(-1)
                else:
                    #grad_norm = torch.linalg.norm(self.model.weight.grad, ord='fro')
                    grad_norm = np.linalg.norm(self.model.weight.grad.detach().numpy())
                    self.gradient_norm_log.append(grad_norm)
                self.optimizer.step()
        #epoch_loss = self.running_epoch_loss / len(trainloader['train'])  # From: epoch_loss = running_epoch_loss / len(dataloaders['train'])
        #self.loss_log.append(epoch_loss)  

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        #if self.privacy:
        #    eps, DELTA = get_dp_params(privacy_engine)
        #    print(f"Client {self.ID}", f"epsilon = {eps:.2f}, sigma = {DELTA}")