
import torch
from sklearn.model_selection import KFold
import numpy as np

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverlocal import Local
from flcore.servers.serverapfl import APFL

from models.DNN_classes import *


def create_user_folds(users, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=42)
    user_folds = list(kf.split(users))
    return user_folds

def init_algo(args):
    if args.algorithm == "FedAvg":
        # FIX ARGS.HEAD --> LINEAR REGRESSION HAS NO FC
        #args.head = copy.deepcopy(args.model.fc)
        #args.model.fc = nn.Identity()
        #args.model = BaseHeadSplit(args.model, args.head)
        server = FedAvg(args)
    elif args.algorithm == "Local":
        server = Local(args)
    elif args.algorithm == "APFL":
        server = APFL(args)
    elif args.algorithm == "PerAvg":
        server = PerAvg(args)
    elif args.algorithm == "pFedMe":
        server = pFedMe(args)
    else:
        raise NotImplementedError
    return server

def init_model(args):
    # Do I need to return args here...

    # Generate args.model
    if args.model_str == "LinearRegression":
        fresh_model_obj = torch.nn.Linear(args.input_size, args.output_size, args.linear_model_bias)  #input_size, output_size, bias boolean
    elif args.model_str == "RNN":
        # Initialize the RNN model
        fresh_model_obj = RNNModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "LSTM":
        # Initialize the LSTM model
        fresh_model_obj.model = LSTMModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "GRU":
        fresh_model_obj = GRUModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "Transformer":
        fresh_model_obj = TransformerModel(args.input_size, args.output_size)
    else:
        raise NotImplementedError
    return fresh_model_obj

def k_fold_cross_validation(self, n_splits=5, epochs=10):
    '''This one is just for reference I think? Dont run/use this code directly'''
    users = self.train_subj_IDs
    user_folds = create_user_folds(users, n_splits)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(user_folds):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Set training and validation users
        self.train_users = [users[i] for i in train_idx]
        self.train_subj_IDs = [users[i].ID for i in train_idx]
        self.train_numerical_subj_IDs = [id_str[-3:] for id_str in self.train_subj_IDs]
        self.val_users = [users[i] for i in val_idx]
        self.val_subj_IDs = [users[i].ID for i in val_idx]
        self.val_numerical_subj_IDs = [id_str[-3:] for id_str in self.val_subj_IDs]
        
        # Initialize a new model for each fold
        self.model = self.init_model()  # Pass in args or do I have self.args?
        
        # Train the model
        for epoch in range(epochs):
            train_loss, train_samples = self.train()  # Your existing train method
            val_loss, val_samples = self.test_metrics()  # Your existing test_metrics method
            
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / train_samples:.4f}, Val Loss: {val_loss / val_samples:.4f}")
        
        # Evaluate on validation set
        final_val_loss, final_val_samples = self.test_metrics()
        cv_results.append(final_val_loss / final_val_samples)
        
        print(f"Fold {fold + 1} Validation Loss: {final_val_loss / final_val_samples:.4f}")
    
    mean_cv_loss = np.mean(cv_results)
    std_cv_loss = np.std(cv_results)
    
    print(f"Cross-validation results: {mean_cv_loss:.4f} (+/- {std_cv_loss:.4f})")
    
    return cv_results, mean_cv_loss, std_cv_loss

#cv_results, mean_cv_loss, std_cv_loss = self.k_fold_cross_validation(n_splits=5, epochs=10)
# cv_results, mean_cv_loss, std_cv_loss = run(args)

def run(args):
    # Need to figure out file saving since it is running 5 times... does it overwrite? 
    # Maybe times is for statistics/repeatability and not just PFL... double check what it is doing and how they used it...

    time_list = []
    reporter = MemReporter()

    server = init_algo(args)
    # Server hasn't even been init'd yet...
    users = args.all_subj_IDs
    user_folds = create_user_folds(users, args.num_kfold_splits)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(user_folds):
        print(f"Fold {fold + 1}/{args.num_kfold_splits}")
        
        # Should probably be accessed as an attribute of the server...
        ## I can actually access it from args I think, just need to match up the var names
        # Set training and validation users
        train_users = [users[i] for i in train_idx]
        val_users = [users[i] for i in val_idx]
        # Where do I need to put this so the server uses it...
        self.testloader = self.load_test_data()
        assert(len(self.testloader)!=0)
        
        # Initialize a new model for each fold
        # RETURNS a fresh model object for args.model!
        args.model = init_model(args)  # Pass in args or do I have self.args?

        # args.times=1 for now... I'm not using this loop at all basically...
        #for i in range(args.prev, args.times):
        #    print(f"\n============= Running time: {i}th =============")
        print(f"\n============= STARTING NEW TRIAL =============")
        start = time.time()
        print(args.model)

        # select algorithm
        # RETURNS server obj!
        if fold!=0:
            server = init_algo(args)

        server.train()
        time_list.append(time.time()-start)

        server.plot_results()
        print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    mean_cv_loss = np.mean(cv_results)
    std_cv_loss = np.std(cv_results)
    print(f"Cross-validation results: {mean_cv_loss:.4f} (+/- {std_cv_loss:.4f})")
      
    # Global average
    if args.algorithm != "Local":
        average_data(server.trial_result_path, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
        # Not super sure what "times" is, by default it is 1. Assuming it runs the process multiple times to average out the stochasticity?

    print("All done!")
    reporter.report()
    return server, cv_results, mean_cv_loss, std_cv_loss


def collect_metrics(self):
    '''Not integrated yet... not sure what else needs to be done'''

    ###############################
    # Idk if this part works... should get subsumed (and ideally not used) by Seq anyways...
    if self.eval_new_clients and self.num_new_clients > 0:
        self.fine_tuning_new_clients()
        return self.test_metrics_new_clients() #collect_metrics_new_clients
    ###############################
    
    train_losses = []
    test_losses = []  # I dont think this should be here... 
    num_samples = []
    num_train_iterations = []
    
    for c in self.clients:
        client_train_loss, client_test_loss, ns, nti = c.get_metrics()
        train_losses.append(client_train_loss)
        test_losses.append(client_test_loss)
        num_samples.append(ns)
        num_train_iterations.append(nti)
    
    ids = [c.id for c in self.clients]
    
    # Calculate average losses
    avg_train_loss = sum([tl * ns for tl, ns in zip(train_losses, num_samples)]) / sum(num_samples)
    avg_test_loss = sum([tl * ns for tl, ns in zip(test_losses, num_samples)]) / sum(num_samples)
    
    return ids, num_samples, num_train_iterations, train_losses, test_losses, avg_train_loss, avg_test_loss

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class UserTimeSeriesDataset(Dataset):
    def __init__(self, data, labels, batch_size=32):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.window_size = batch_size

    def __len__(self):
        return len(self.data) - self.batch_size + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.batch_size], self.labels[idx:idx+self.batch_size]

def create_crossval_test_dataloader(user_datasets, user_labels, batch_size=32):
    datasets = [UserTimeSeriesDataset(user_data, user_label) 
                for user_data, user_label in zip(user_datasets, user_labels)]
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

# Example usage
#user_datasets = [np.random.randn(20770, 64) for _ in range(5)]  # 5 users, each with [20770, 64] data
#user_labels = [np.random.randn(20770, 2) for _ in range(5)]  # 5 users, each with [20770, 2] labels
#test_dataloader = create_crossval_test_dataloader(user_datasets, user_labels)