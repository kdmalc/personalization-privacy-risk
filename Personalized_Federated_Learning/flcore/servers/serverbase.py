# PFLNIID

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
#import pickle
from datetime import datetime

from flcore.pflniid_utils.data_utils import read_client_data
#from flcore.pflniid_utils.dlg import DLG
#from utils import node_creator


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
        # Can't save dicts to HD5F files...
        #self.cost_func_comps_dict = dict()
        #self.gradient_dict = dict()
        self.cost_func_comps_log = []
        self.gradient_norm_log = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        #self.dlg_eval = args.dlg_eval
        #self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        
        # Kai's additional params
        self.global_round = 0
        self.test_split_fraction = args.test_split_fraction
        self.condition_number = args.condition_number
        self.debug_mode = args.debug_mode
        self.global_update = args.starting_update
        # No idea what the point of this was...
        #if self.debug_mode:
        #    self.all_user_keys = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']
        #    if self.dataset.upper()=='CPHS':
        #        with open(r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\Data\continuous_full_data_block1.pickle", 'rb') as handle:
        #            self.all_labels, _, _, _, self.all_emg, _, _, _, _, _, _ = pickle.load(handle)
        #    else:
        #        raise ValueError("Dataset not supported")
        self.test_split_each_update = args.test_split_each_update
        self.verbose = args.verbose
        self.slow_clients_bool = args.slow_clients_bool

    def set_clients(self, clientObj):  
        if self.verbose:
            print("ServerBase Set_Clients (SBSC) -- probably called in init() of server children classes")
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            print(f"SBSC: iter {i}")
            # Should I switch i to be the subject ID? Not specifically required to run, for now at least
            # ID = i probably isn't the best solution... assumes things are in order...... no? Not a good solution regardless
            base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Data\\Client_Specific_Files\\'
            client = clientObj(self.args, 
                                ID=i, 
                                train_samples = base_data_path + "UserID" + str(i) + "_TrainData_8by20770by64.npy", 
                                test_samples = base_data_path + "UserID" + str(i) + "_Labels_8by20770by2.npy", 
                                train_slow=train_slow, 
                                send_slow=send_slow)
            
            self.clients.append(client)
            

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        # I never updated this but it is run in serverlocal for some reason idk
        #raise ValueError("select_slow_clients() has not been updated yet.")
        slow_clients = [False for i in range(self.num_clients)]
        if self.slow_clients_bool==False:
            return slow_clients
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        # I never updated this but it is run in serverlocal for some reason idk
        #print("set_slow_clients has been run")
        #raise ValueError("set_slow_clients() has not been updated yet.")
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
        print("SENDING GLOBAL MODEL TO CLIENTS")
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
                tot_samples += client.train_samples  # tot_samples += client.num_train_samples
                self.uploaded_IDs.append(client.ID)
                self.uploaded_weights.append(client.train_samples)  # What is going on here
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
        
    def save_results(self, save_cost_func_comps=False, save_gradient=False):
        algo = self.dataset + "_" + self.algorithm

        # Is this path wrt serverbase.py or main.py...
        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        str_current_datetime = str(current_datetime)

        result_path = "../results/mdHM_" + str_current_datetime + "_"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_train_loss)):  # rs_train_acc
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if save_cost_func_comps:
                    print("cost_func_comps_log")
                    print(self.cost_func_comps_log)
                    print()                    
                    G1 = hf.create_group('cost_func_tuples_by_client')
                    for idx, cost_func_comps in enumerate(self.cost_func_comps_log):
                        name_str = 'ClientID' + str(idx)
                        G1.create_dataset(name_str, data=cost_func_comps)
                
                if save_gradient:
                    print('gradient_norm_log')
                    print(self.gradient_norm_log)
                    print()
                    G2 = hf.create_group('gradient_norm_lists_by_client')
                    for idx, grad_norm_list in enumerate(self.gradient_norm_log):
                        name_str = 'ClientID' + str(idx)
                        G2.create_dataset(name_str, data=grad_norm_list)

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
        for c in self.clients:
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
        for c in self.clients:
            if self.verbose:
                print(f"Serverbase train_metrics(): Client{c.ID}")
            c.last_global_round = self.global_round
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, losses

    # evaluate selected clients
    def evaluate(self, train=True, test=True, acc=None, loss=None):
        '''
        KAI Docstring
        This func runs test_metrics and train_metrics, and then sums all of
        Previously, test_metrics and train_metrics were collecting the losses on ALL clients (even the untrained ones...)
        I switched that (5/31 12:06pm) to be just the selected clients, the idea being that ALL clients explode the loss func
        '''
        if self.verbose:
            print("Serverbase evaluate()")
        if test:
            stats = self.test_metrics()
            if self.verbose:
                print(f"Len of test_metrics() output: {len(stats[0])}")
            #test_loss = sum(stats[2])*1.0 / len(stats[2])  # Idk what this was doing either. Not relevant to us...
            #test_loss = sum(stats[2])*1.0  # Used to return test_acc, test_num, auc; idk what it is summing tho (or why auc wouldn't be a scalar...)
            test_loss = stats[2]#*1.0  #It's already a float...

            if acc == None:
                # Idk what rs is...
                avg_test_loss = sum(test_loss)/len(test_loss)
                self.rs_test_loss.append(avg_test_loss)
            else:
                acc.append(test_loss)

            #assert(test_loss<1e5)
            print("Averaged Test Loss: {:.4f}".format(avg_test_loss))

        if train:
            stats_train = self.train_metrics()
            if self.verbose:
                print(f"Len of train_metrics() output: {len(stats_train[0])}")
            #train_loss = sum(stats_train[2])*1.0
            #train_loss = sum(stats_train[2])*1.0 / len(stats_train[2])
            train_loss = stats_train[2]#*1.0
        
            if loss == None:
                avg_train_loss = sum(train_loss)/len(train_loss)
                self.rs_train_loss.append(avg_train_loss)
            else:
                print("Server evaluate loss!=None!")
                loss.append(train_loss)

            #assert(train_loss<1e5)
            print("Averaged Train Loss: {:.4f}".format(avg_train_loss))

        # I don't think I still need this...
        assert(type(self.rs_train_loss)==type(list()))


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        print("Running check_done")
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
        raise ValueError("call_dlg has not been developed yet")
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
        #print("---------------> Serverbase set_new_clients: still using read_client_data and lengths...")
        #for i in range(self.num_clients, self.num_clients + self.num_new_clients):
        #    train_data = read_client_data(self.dataset, i, self.global_update, is_train=True)
        #   test_data = read_client_data(self.dataset, i, self.global_update, is_train=False)
        #    client = clientObj(self.args, 
        #                    ID=i, 
        #                    train_samples=len(train_data), 
        #                    test_samples=len(test_data), 
        #                    train_slow=False, 
        #                    send_slow=False)
        #    self.new_clients.append(client)
        #if self.verbose:
        #    print("ServerBase set_new_clients (SBSNC)")
        if self.num_new_clients==0:
            pass
        else:
            assert("set_new_clients must be refactored, IDs by index will not work here")
            #for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            for i in range(self.num_clients, self.num_clients + self.num_new_clients):
                print(f"SBSNC: iter {i}")
                # Should I switch i to be the subject ID? Not specifically required to run, for now at least
                # ID = i probably isn't the best solution... assumes things are in order...... no? Not a good solution regardless
                base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Data\\Client_Specific_Files\\'
                client = clientObj(self.args, 
                                    ID=i, 
                                    train_samples = base_data_path + "UserID" + str(i) + "_TrainData_8by20770by64.npy", 
                                    test_samples = base_data_path + "UserID" + str(i) + "_Labels_8by20770by2.npy", 
                                    train_slow=False, 
                                    send_slow=False)
                self.clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        print("fine_tuning_new_clients USES GLOBAL MODEL!!!")
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
