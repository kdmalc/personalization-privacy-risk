# PFLNIID

import torch
import os
import numpy as np
import h5py
import copy
import time
import random

from flcore.pflniid_utils.data_utils import read_client_data
from flcore.pflniid_utils.dlg import DLG
from utils import node_creator


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold  # Spelled wrong lol
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 20
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_IDs = []
        self.uploaded_models = []

        self.rs_test_loss = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        
        # Kai's additional params
        self.global_round = 0
        self.test_split = args.test_split
        self.condition_number = args.condition_number
        self.debug_mode = args.debug_mode
        if self.debug_mode:
            self.all_user_keys = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']
            if self.dataset.upper()=='CPHS':
                with open(r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\Data\continuous_full_data_block1.pickle", 'rb') as handle:
                    self.all_labels, _, _, _, self.all_emg, _, _, _, _, _, _ = pickle.load(handle)
            else:
                raise("Dataset not supported")

    def set_clients(self, clientObj):  
        print("Serverbase set_clients")
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            print(f"SBSC: iter {i}")
            # Should I switch i to be the subject ID? Not required idk
            if self.debug_mode:
                print("DEBUG MODE")
                
                # This assumes that the id's are in order.
                # This is fine when using all clients, otherwise would need to map idx to the included subjects' IDs
                upper_bound = round(self.test_split*(self.all_emg[self.all_user_keys[i]][self.condition_number,:,:].shape[0]))
                train_data = self.all_emg[self.all_user_keys[i]][self.condition_number,:upper_bound,:]
                test_data = self.all_emg[self.all_user_keys[i]][self.condition_number,upper_bound:,:]
                
                # So where do I actually give the client their data?
                #CustomEMGDataset(emgs_block1[my_user][condition_number,:upper_bound,:], refs_block1[my_user][condition_number,:upper_bound,:])
            else:
                print("Setting train_data")
                train_data = read_client_data(self.dataset, i, is_train=True)
                print("Setting test_data")
                test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            ID=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
            
    def _set_clients(self, clientObj):
        print("_set_clients(): I haven't edited this one yet really")
        
        # Still under development
        dataset_list = make_users(condition_number=self.condition_number, dataset=self.dataset)
        
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # Should I switch i to be the subject ID? Not required idk
            
            # Should I revamp or replace read_client_data?
            ## Would be nice to revamp it...
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            
            # This is all fine
            ## Although I think all clients get the same args but whatever
            client = clientObj(self.args, 
                            ID=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        self.uploaded_IDs = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_IDs.append(client.ID)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_train_loss)):  # rs_train_acc
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                #hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                #hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_loss = []
        # Switching to just testing on the selected clients
        for c in self.selected_clients:  #self.clients:
            tl, ns = c.test_metrics()
            tot_loss.append(tl*1.0)
            num_samples.append(ns)

        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, tot_loss

    def train_metrics(self):
        self.global_round += 1
        
        if self.eval_new_clients and self.num_new_clients > 0:
            print("KAI: Returned early for some reason, idk what this code is doing")
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        print(f"Serverbase train_metrics(): GLOBAL ROUND: {self.global_round}")
        # Switching to just testing on the selected clients
        for c in self.selected_clients:  #self.clients:
            print(f"Serverbase train_metrics(): Client{c.ID}")
            c.last_global_round = self.global_round
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        '''
        KAI Docstring
        This func runs test_metrics and train_metrics, and then sums all of
        Previously, test_metrics and train_metrics were collecting the losses on ALL clients (even the untrained ones...)
        I switched that (5/31 12:06pm) to be just the selected clients, the idea being that ALL clients explode the loss func
        '''
        print("Serverbase evaluate()")
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        # Should these be divided by something?
        # Do we not have a train_loss for every training round?...
        #test_loss = sum(stats[2])*1.0
        #train_loss = sum(stats_train[2])*1.0
        # Dividing by the length (should it be num samples instead...)
        print(f"Len of test_metrics() output: {len(stats[2])}")
        print(f"Len of train_metrics() output: {len(stats_train[2])}")
        test_loss = sum(stats[2])*1.0 / len(stats[2])
        train_loss = sum(stats_train[2])*1.0 / len(stats_train[2])
        
        if acc == None:
            self.rs_test_loss.append(test_loss)
        else:
            acc.append(test_loss)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        assert(np.isnan(train_loss)==False)
        assert(np.isnan(test_loss)==False)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Loss: {:.4f}".format(test_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    # No idea what this does, I don't think it is used...
    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cID, client_model in zip(self.uploaded_IDs, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cID].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break
                    
                    # Some CUDA stuff to ignore for now
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        print("Serverbase set_new_clients")
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            ID=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            for e in range(self.fine_tuning_epoch):
                client.train()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_loss = []
        for c in self.clients:
            tl, ns = c.test_metrics()
            tot_loss.append(tl*1.0)
            num_samples.append(ns)

        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, tot_loss
