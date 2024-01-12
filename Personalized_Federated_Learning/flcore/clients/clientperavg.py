import numpy as np
import torch
import time
import copy
#import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client
#from utils.data_utils import read_client_data
#from torch.utils.data import DataLoader


class clientPerAvg(Client):
    def __init__(self, args, ID, samples_path, labels_path, condition_number, **kwargs):
        super().__init__(args, ID, samples_path, labels_path, condition_number, **kwargs)

        self.beta = args.beta
        # self.learning_rate = args.local_learning_rate --> This is set in clientbase already (inherited)

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        # Does this imply that it is on no matter what? I believe so...
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        # trainloader obj will have a bs twice that of the input one, effectively sampling twice as many points
        ## So that way you can run step 1 and step 2 from PerFedAvg
        ### This is FO MAML...
        trainloader = self.load_train_data(self.batch_size*2)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(self.local_epochs):  # local update
            for X, Y in trainloader:
                # Use list because self.model.parameters() is an iterator 
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # How to integrate X,Y, x,y, and self.F?
                # step 1
                if type(X) == type([]):
                    print("X is a list for some reason idk")
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                #self.simulate_data_streaming_xy(x, y) #--> Already called in shared_loss_calc() below
                #if self.train_slow:
                #    time.sleep(0.1 * np.abs(np.random.rand()))

                #output = self.model(x)
                #loss = self.loss(output, y)
                loss, num_samples = self.shared_loss_calc(x, y, self.model, record_cost_func_comps=True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # step 2
                # How is this any different from step 1?
                ## I'm assuming this is the local fine-tuning or smth or other?
                if type(X) == type([]):
                    print("X is a list still/again?")
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)

                #self.simulate_data_streaming_xy(x, y) #--> Called in shared_loss_calc() below, still
                #if self.train_slow:
                #    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                #output = self.model(x)
                #loss = self.loss(output, y)
                loss, num_samples = self.shared_loss_calc(x, y, self.model, record_cost_func_comps=False) # Idk if it should be recording or not...
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()

        (x, y) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)

        #self.simulate_data_streaming_xy(x, y)
        #output = self.model(x)
        #loss = self.loss(output, y)
        loss, num_samples = self.shared_loss_calc(x, y, self.model, record_cost_func_comps=False) # Idk if it should be recording or not here...
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.model.cpu()


    def train_metrics(self, saved_model_path=None, model_obj=None):
        trainloader = self.load_train_data(batch_size=self.batch_size*2, eval=True)
        if model_obj != None:
            eval_model = model_obj
        elif saved_model_path != None:
            eval_model = self.load_model(saved_model_path)
        else:
            eval_model = self.model
        eval_model.eval()

        train_num = 0
        losses = 0
        for X, Y in trainloader:
            # step 1
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][:self.batch_size].to(self.device)
                x[1] = X[1][:self.batch_size]
            else:
                x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)

            self.optimizer.zero_grad()
            #output = self.model(x)
            #loss = self.loss(output, y)
            loss, num_samples = self.shared_loss_calc(x, y, eval_model, record_cost_func_comps=False) # Idk if it should be recording or not here...
            loss.backward()
            self.optimizer.step()

            # step 2
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][self.batch_size:].to(self.device)
                x[1] = X[1][self.batch_size:]
            else:
                x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)
            
            self.optimizer.zero_grad()
            #output = self.model(x)
            #loss1 = self.loss(output, y)
            loss1, num_samples = self.shared_loss_calc(x, y, eval_model, record_cost_func_comps=False) # Idk if it should be recording or not here...

            train_num += y.shape[0]
            #print(f"CPerAvg train_metrics y.shape: {y.shape[0]}, x.shape: {x.shape[0]}")
            losses += loss1.item() * y.shape[0]
        #print()
        # This would be where the loss_log.append(losses / train_num) would be... this happens in evaluate() tho...

        return losses, train_num

    def train_one_epoch(self):
        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            #output = self.model(x)
            #loss = self.loss(output, y)
            loss, num_samples = self.shared_loss_calc(x, y, self.model, record_cost_func_comps=False) # Idk if it should be recording or not here...
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # self.model.cpu()