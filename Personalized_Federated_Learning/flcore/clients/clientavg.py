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
            for i, (x, y) in enumerate(trainloader):  # i currently have it set such that each tl only has 1 batch of 1200 (8/5/23)
                # Put/call def simulate_data_stream here?
                self.simulate_data_streaming_xy(x, y)

                # Idk if this needs to happen if I'm just running it on cpu...
                # ^ If so it would need to happen in simulate data
                #if type(x) == type([]):
                #    x[0] = x[0].to(self.device)
                #else:
                #    x = x.to(self.device)
                #y = y.to(self.device)
                #if self.train_slow:
                #    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # reset gradient so it doesn't accumulate
                self.optimizer.zero_grad()
                # forward pass and loss
                # D@s = predicted velocity
                vel_pred = self.model(torch.transpose(self.F, 0, 1)) 
                
                if vel_pred.shape[0]!=self.y_ref.shape[0]:
                    tvel_pred = torch.transpose(vel_pred, 0, 1)
                else:
                    tvel_pred = vel_pred
                t1 = self.loss_func(tvel_pred, self.y_ref)
                t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model.weight))**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                #detach().numpy()
                if np.isnan(t1.item()):
                    raise ValueError("CLIENTAVG: Error term is NAN...")
                if np.isnan(t2.item()):
                    raise ValueError("CLIENTAVG: Decoder Effort term is NAN...")
                if np.isnan(t3.item()):
                    raise ValueError("CLIENTAVG: User Effort term is NAN...")
                loss = t1 + t2 + t3
                self.cost_func_comps_log = [(t1.item(), t2.item(), t3.item())]
                
                # backward pass
                loss.backward(retain_graph=True)
                self.loss_log.append(loss.item())
                # update weights
                self.optimizer.step()

                # This would need to be changed if you switch to a sequential (not single layer) model
                # Gradient norm
                weight_grad = self.model.weight.grad
                if weight_grad == None:
                    print("Weight gradient is None...")
                    self.gradient_norm_log.append(-1)
                else:
                    #grad_norm = torch.linalg.norm(self.model.weight.grad, ord='fro') 
                    # ^Equivalent to the below but its still a tensor
                    grad_norm = np.linalg.norm(self.model.weight.grad.detach().numpy())
                    self.gradient_norm_log.append(grad_norm)

                print(f"Step {step}, pair {i} in traindl; x.size(): {x.size()}; loss: {loss.item():0.2f}")
                #self.running_epoch_loss.append(loss.item() * x.size(0))  # From: running_epoch_loss.append(loss.item() * images.size(0))
                #running_num_samples += x.size(0)

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