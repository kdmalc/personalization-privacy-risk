# PFLNIID

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from datetime import datetime
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
        self.local_learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.personalized_algorithms = ["APFL", "FedMTL", "PerAvg", "pFedMe", "FedPer", "Ditto"]
        self.personalized_algo_bool = True if self.algorithm.upper() in [algo.upper() for algo in self.personalized_algorithms] else False
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threshold = args.time_threshold
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
        # Can't save dicts to HD5F files so use nested lists for now I guess
        self.cost_func_comps_log = []
        self.gradient_norm_log = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.new_clients_ID_lst = args.new_clients_ID_lst
        self.num_new_clients = len(self.new_clients_ID_lst)
        self.new_clients_obj_lst = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        
        # Kai's additional params
        self.ndp = args.num_decimal_points
        self.global_round = 0
        self.test_split_fraction = args.test_split_fraction
        self.debug_mode = args.debug_mode
        self.global_update = args.starting_update
        self.verbose = args.verbose
        self.slow_clients_bool = args.slow_clients_bool
        self.run_train_metrics = args.run_train_metrics
        self.result_path = r"C:\Users\kdmen\Desktop\Research\personalization-privacy-risk\Personalized_Federated_Learning\results\mdHM_" 
        # Not used on server but saved when logging
        self.pca_channels = args.pca_channels
        self.device_channels = args.device_channels
        self.lambdaF = args.lambdaF
        self.lambdaD = args.lambdaD
        self.lambdaE = args.lambdaE
        self.normalize_data = args.normalize_data
        self.local_round_threshold = args.local_round_threshold
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        # Testing
        self.test_split_users = args.test_split_users
        self.test_split_each_update = args.test_split_each_update
        self.test_subj_IDs = args.test_subj_IDs
        # Trial set up
        self.condition_number_lst = args.condition_number_lst
        self.train_subj_IDs = args.train_subj_IDs
        self.num_clients = len(self.train_subj_IDs) * len(self.condition_number_lst)
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        # This might need to get changed, but works as long as last 3 chars are the numerical subject ID (eg we drop the header and 'S')
        self.train_numerical_subj_IDs = [id_str[-3:] for id_str in self.train_subj_IDs]
        self.loss_threshold = args.loss_threshold
        ## SEQUENTIAL TRAINING PARAMS
        self.sequential = args.sequential
        self.live_clients = args.live_clients
        self.static_clients = args.static_clients
        self.static_vs_live_weighting = args.static_vs_live_weighting
        self.prev_model_directory = args.prev_model_directory


    def set_clients(self, clientObj):  
        if self.verbose:
            print("ServerBase Set_Clients (SBSC) -- probably called in init() of server children classes")
        
        base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Data\\Subject_Specific_Files\\'
        for i, train_slow, send_slow in zip(range(len(self.train_subj_IDs)), self.train_slow_clients, self.send_slow_clients):
            for j in self.condition_number_lst:
                print(f"SB Set Client: iter {i}, cond number: {str(j)}")
                client = clientObj(self.args, 
                                    ID=self.train_subj_IDs[i], 
                                    samples_path = base_data_path + 'S' + str(self.train_numerical_subj_IDs[i]) + "_TrainData_8by20770by64.npy", 
                                    labels_path = base_data_path + 'S' + str(self.train_numerical_subj_IDs[i]) + "_Labels_8by20770by2.npy", 
                                    condition_number = j-1, 
                                    train_slow=train_slow, 
                                    send_slow=send_slow)
                self.clients.append(client)
                if self.sequential:
                    # NOT FINISHED YET 
                    
                    # Load the client model
                    full_path_to_item = self.prev_model_directory + self.ID
                    client.load_item(None, full_path_to_item)
                else:
                    client.load_train_data(client_init=True)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for _ in range(self.num_clients)]
        if self.slow_clients_bool==False:
            return slow_clients
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
            if client_time_cost <= self.time_threshold:
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

    '''
    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
    '''

    def load_model(self, directory_name, type):
        '''
            Loads the specified model to become the current global model!

            str directory_name: name of when the model was saved, likely in the form of %m-%d_%H-%M unless it was renamed (this is the model's directory)
            str type: Should be one of 'global', 'pers', or 'local'
        '''
        model_path = os.path.join("models", self.dataset, directory_name, self.algorithm + "_server_" + type + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self, directory_name, type):
        '''
            str directory_name: name of when the model was saved, likely in the form of %m-%d_%H-%M unless it was renamed (this is the model's directory)
            str type: Should be one of 'global', 'pers', or 'local'
        '''
        model_path = os.path.join("models", self.dataset, directory_name, self.algorithm + "_server_" + type + ".pt")
        return os.path.exists(model_path)
    
        
    def save_results(self, personalized=False, save_cost_func_comps=False, save_gradient=False):
        algo = self.dataset + "_" + self.algorithm

        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        str_current_datetime = str(current_datetime)

        self.trial_result_path = self.result_path + str_current_datetime
        if not os.path.exists(self.trial_result_path):
            os.makedirs(self.trial_result_path)
        model_path = os.path.join("models", self.dataset, str_current_datetime)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server_global.pt")
        torch.save(self.global_model, model_path)
        if personalized==True:
            pers_model_path = os.path.join(model_path, self.algorithm + "_client_pers_model")
            for c in self.clients:
                if not os.path.exists(pers_model_path):
                    print(f"SB pers model save made directory! {pers_model_path}")
                    os.makedirs(pers_model_path)
                torch.save(c.model, os.path.join(model_path, self.algorithm + "_client_pers_model", c.ID + "_pers_model.pt"))

        param_log_str = (
            "BASE\n"
            f"algorithm = {self.algorithm}\n"
            f"model = {self.global_model}\n"
            f"train_subj_IDs = {self.train_subj_IDs}\n"
            f"condition_number_lst = {self.condition_number_lst}\n"
            f"total effective clients = train_subj_IDs*condition_number_lst = {self.num_clients}\n"
            f"device_channels = {self.device_channels}\n"
            "\nMODEL HYPERPARAMETERS\n"
            f"lambdaF = {self.lambdaF}\n"
            f"lambdaD = {self.lambdaD}\n"
            f"lambdaE = {self.lambdaE}\n"
            f"global_rounds = {self.global_rounds}\n"
            f"local_epochs = {self.local_epochs}\n"
            f"batch_size = {self.batch_size}\n"
            f"local_learning_rate = {self.local_learning_rate}\n"
            f"learning_rate_decay = {self.learning_rate_decay}\n"
            f"learning_rate_decay_gamma = {self.learning_rate_decay_gamma}\n"
            f"pca_channels = {self.pca_channels}\n"
            f"normalize_data = {self.normalize_data}\n"
            "\nFEDERATED LEARNING PARAMS\n"
            f"starting_update = {self.args.starting_update}\n"
            f"local_round_threshold = {self.local_round_threshold}\n"
            "\nTESTING\n"
            f"test_split_fraction = {self.test_split_fraction}\n"
            f"test_split_each_update = {self.test_split_each_update}\n"
            f"test_split_users = {self.test_split_users}\n"
            f"run_train_metrics = {self.run_train_metrics}")
        with open(self.trial_result_path+r'\param_log.txt', 'w') as file:
            file.write(param_log_str)

        if (personalized==True and ((len(self.rs_test_loss_per))!=0)) or (personalized==False and ((len(self.rs_test_loss))!=0)):
            # Why would this run if run_train_metrics is False...
            algo = algo + "_" + self.goal# + "_" + str(self.times)  # IDk what self.times represents...
            file_path = self.trial_result_path + r"\{}.h5".format(algo)
            print("File path: " + file_path)
            #for client in self.clients:
            #    client.results_file_path = self.trial_result_path
            #    client.h5_file_path = file_path

            with h5py.File(file_path, 'w') as hf:
                if personalized:
                    hf.create_dataset('rs_test_loss_per', data=self.rs_test_loss_per)
                    hf.create_dataset('rs_train_loss_per', data=self.rs_train_loss_per)
                else:
                    hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if save_cost_func_comps:
                    #print(f'cost_func_comps_log: \n {self.cost_func_comps_log}\n')                   
                    G1 = hf.create_group('cost_func_tuples_by_client')
                    for idx, cost_func_comps in enumerate(self.cost_func_comps_log):
                        name_index = idx // len(self.condition_number_lst)
                        if name_index >= len(self.train_subj_IDs):
                            name_index = len(self.train_subj_IDs) - 1  # Ensure it doesn't exceed the last index
                        name_str = self.train_subj_IDs[name_index] + "_C" + str(self.condition_number_lst[idx%len(self.condition_number_lst)])
                        G1.create_dataset(name_str, data=cost_func_comps)
                if save_gradient:
                    #print(f'gradient_norm_log: \n {self.gradient_norm_log}\n')
                    G2 = hf.create_group('gradient_norm_lists_by_client')
                    for idx, grad_norm_list in enumerate(self.gradient_norm_log):
                        name_index = idx // len(self.condition_number_lst)
                        if name_index >= len(self.train_subj_IDs):
                            name_index = len(self.train_subj_IDs) - 1  # Ensure it doesn't exceed the last index
                        name_str = self.train_subj_IDs[name_index] + "_C" + str(self.condition_number_lst[idx%len(self.condition_number_lst)])
                        G2.create_dataset(name_str, data=grad_norm_list)


    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            print(f"SB save_item() made directory! {self.save_folder_name}")
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
            # if eval new client AND we actually have new clients, then return this
            # maps to depreceated values I think...
            print("KAI: Returned early for some reason, idk what this code is doing")
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        print(f"Serverbase train_metrics(): GLOBAL ROUND: {self.global_round}")
        for c in self.clients:
            if self.verbose:
                print(f"Serverbase train_metrics(): {c.ID}")
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
                avg_test_loss = sum(test_loss)/len(test_loss)
                self.rs_test_loss.append(avg_test_loss)
            else:
                acc.append(test_loss)

            #assert(test_loss<1e5)
            print("Averaged Test Loss: {:.5f}".format(avg_test_loss))

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

            print("Averaged Train Loss: {:.5f}".format(avg_train_loss))
            if avg_train_loss>self.loss_threshold:
                # Log training loss up to this point...
                self.save_results(personalized=self.personalized_algo_bool, save_cost_func_comps=True, save_gradient=True)
                raise ValueError('Averaged training loss exceeded the maximum loss threshold, aborting training.')


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
        if self.num_new_clients==0:
            pass
        else:
            base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Data\\Subject_Specific_Files\\'
            for i in range(self.num_clients, self.num_clients + self.num_new_clients):
                # Idk I guess I can keep the condition iter? Idk why I would want to turn it off other than not expecting it
                for j in self.condition_number_lst:
                    print(f"SB Set New Client: iter iter {i}, cond number: {str(j)}")
                    client = clientObj(self.args, 
                                        ID=self.train_subj_IDs[i], 
                                        samples_path = base_data_path + 'S' + str(self.train_numerical_subj_IDs[i]) + "_TrainData_8by20770by64.npy", 
                                        labels_path = base_data_path + 'S' + str(self.train_numerical_subj_IDs[i]) + "_Labels_8by20770by2.npy", 
                                        condition_number = j-1, 
                                        train_slow=False, 
                                        send_slow=False)
                    self.clients.append(client)
                    #client.load_test_data(client_init=True)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        print("fine_tuning_new_clients USES GLOBAL MODEL!!!")
        for client in self.new_clients_obj_lst:
            client.set_parameters(self.global_model)
            for _ in range(self.fine_tuning_epoch):
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
    