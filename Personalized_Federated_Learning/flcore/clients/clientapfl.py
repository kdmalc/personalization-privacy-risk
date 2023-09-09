# From PFL-Non-IID

import copy
import torch
#import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client

from sklearn import metrics

class clientAPFL(Client):
    def __init__(self, args, ID, samples_path, labels_path, condition_number, **kwargs):
        super().__init__(args, ID, samples_path, labels_path, condition_number, **kwargs)

        self.alpha = args.alpha
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        self.model_per.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                print(f"Step {step}, batch {i}")
                self.cphs_training_subroutine(x, y)

                output_per = self.model_per(torch.transpose(self.F, 0, 1))
                if output_per.shape[0]!=self.y_ref.shape[0]:
                    output_per = torch.transpose(output_per, 0, 1)
                t1 = self.loss_func(output_per, self.y_ref)
                t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model_per.weight))**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                loss_per = t1 + t2 + t3
                self.optimizer_per.zero_grad()
                loss_per.backward()
                self.optimizer_per.step()

                self.alpha_update()

        for lp, p in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = (1 - self.alpha) * p + self.alpha * lp

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.model.parameters(), self.model_per.parameters()):
            dif = p_params.data - l_params.data
            grad = self.alpha * p_params.grad.data + (1-self.alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))
        
        grad_alpha += 0.02 * self.alpha
        self.alpha = self.alpha - self.learning_rate * grad_alpha
        self.alpha = np.clip(self.alpha.item(), 0.0, 1.0)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model_per.eval()

        running_test_loss = 0
        num_samples = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(testloaderfull):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.simulate_data_streaming_xy(x, y)
                # D@s = predicted velocity
                output = self.model_per(torch.transpose(self.F, 0, 1))
                if output.shape[0]!=self.y_ref.shape[0]:
                    output = torch.transpose(output, 0, 1)
                t1 = self.loss_func(output, self.y_ref)
                t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model_per.weight))**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                loss = t1 + t2 + t3
                test_loss = loss.item()  # Just get the actual loss function term
                running_test_loss += test_loss
                if self.verbose:
                    print(f"batch {i}, loss {test_loss:0,.5f}")
                num_samples += x.size()[0]
        return running_test_loss, num_samples
        

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model_per.train()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.simulate_data_streaming_xy(x, y)
                # D@s = predicted velocity
                output_per = self.model_per(torch.transpose(self.F, 0, 1))
                if output_per.shape[0]!=self.y_ref.shape[0]:
                    output_per = torch.transpose(output_per, 0, 1)
                t1 = self.loss_func(output_per, self.y_ref)
                t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model_per.weight))**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                loss_per = t1 + t2 + t3
                train_num += y.shape[0]
                losses += loss_per.item() * y.shape[0]

        return losses, train_num