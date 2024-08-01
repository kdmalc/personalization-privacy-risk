import numpy as np
import time
import copy
import torch
#import torch.nn as nn
from flcore.optimizers.fedoptimizer import pFedMeOptimizer
from flcore.clients.clientbase import Client


class clientpFedMe(Client):
    def __init__(self, args, ID, samples_path, labels_path, **kwargs):
        super().__init__(args, ID, samples_path, labels_path, **kwargs)

        self.lamda = args.lamda
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate

        # these parameters are for personalized federated learing.
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = pFedMeOptimizer(
            self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):  # local update
            for i, (x, y) in enumerate(trainloader):
                print(f"Step {step}, batch {i}")
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.assert_tl_samples_match_npy(x, y)
                # Simulate datastreaming, eg set s, F and V
                self.simulate_data_streaming_xy(x, y)
                
                # K is number of personalized steps
                for i in range(self.K):
                    # forward pass and loss
                    vel_pred = self.model(torch.transpose(self.F, 0, 1)) 
                    if vel_pred.shape[0]!=self.y_ref.shape[0]:
                        tvel_pred = torch.transpose(vel_pred, 0, 1)
                    else:
                        tvel_pred = vel_pred
                    t1 = self.loss_func(tvel_pred, self.y_ref)
                    t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model.weight))**2)
                    t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                    loss = t1 + t2 + t3
                    self.cost_func_comps_log = [(t1.item(), t2.item(), t3.item())]
                    self.optimizer.zero_grad()
                    # backward pass
                    loss.backward()
                    self.loss_log.append(loss.item())
                    weight_grad = self.model.weight.grad
                    if weight_grad == None:
                        self.gradient_norm_log.append(-1)
                    else:
                        grad_norm = np.linalg.norm(self.model.weight.grad.detach().numpy())
                        self.gradient_norm_log.append(grad_norm)
                    # finding aproximate theta
                    self.personalized_params = self.optimizer.step(self.local_params, self.device)

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.update_parameters(self.model, self.local_params)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        running_test_loss = 0
        test_num = 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(testloaderfull):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                self.simulate_data_streaming_xy(x, y)
                # D@s = predicted velocity
                vel_pred = self.model(torch.transpose(self.F, 0, 1)) 
                
                if vel_pred.shape[0]!=self.y_ref.shape[0]:
                    #print("TRANSPOSING")
                    tvel_pred = torch.transpose(vel_pred, 0, 1)
                else:
                    tvel_pred = vel_pred
                t1 = self.loss_func(tvel_pred, self.y_ref)
                t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model.weight))**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                loss = t1 + t2 + t3

                #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_loss = loss.item()  # Just get the actual loss function term
                running_test_loss += test_loss
                test_num += y.shape[0]

                if self.verbose:
                    print(f"batch {i}, loss {test_loss:0,.5f}")
        # self.model.cpu()
        return running_test_loss, test_num

    def train_metrics_personalized(self):
        trainloader = self.load_train_data(eval=True)
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.simulate_data_streaming_xy(x, y)
                vel_pred = self.model(torch.transpose(self.F, 0, 1)) 
                if vel_pred.shape[0]!=self.y_ref.shape[0]:
                    tvel_pred = torch.transpose(vel_pred, 0, 1)
                else:
                    tvel_pred = vel_pred
                t1 = self.loss_func(tvel_pred, self.y_ref)
                t2 = self.lambdaD*(torch.linalg.matrix_norm((self.model.weight))**2)
                t3 = self.lambdaF*(torch.linalg.matrix_norm((self.F))**2)
                loss = (t1 + t2 + t3).item()
                if self.verbose:
                    print(f"batch {i}, loss {loss:0,.5f}")

                lm = torch.cat([p.data.view(-1) for p in self.local_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.personalized_params], dim=0)
                loss += 0.5 * self.lamda * torch.norm(lm-pm, p=2)

                #train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        # self.model.cpu()
        return losses, train_num
