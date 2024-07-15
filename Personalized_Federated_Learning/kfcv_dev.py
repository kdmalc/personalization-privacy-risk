
import torch
from sklearn.model_selection import KFold
import numpy as np


def create_user_folds(users, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    user_folds = list(kf.split(users))
    return user_folds

def init_algo(self, args):
    if args.algorithm == "FedAvg":
        # FIX ARGS.HEAD --> LINEAR REGRESSION HAS NO FC
        #args.head = copy.deepcopy(args.model.fc)
        #args.model.fc = nn.Identity()
        #args.model = BaseHeadSplit(args.model, args.head)
        server = FedAvg(args, i)
    elif args.algorithm == "Local":
        server = Local(args, i)
    elif args.algorithm == "APFL":
        server = APFL(args, i)
    elif args.algorithm == "PerAvg":
        server = PerAvg(args, i)
    elif args.algorithm == "pFedMe":
        server = pFedMe(args, i)
    else:
        raise NotImplementedError
    return server

def init_model(self, args):
    # Do I need to return args here...

    # Generate args.model
    if args.model_str == "LinearRegression":
        args.model = torch.nn.Linear(args.input_size, args.output_size, args.linear_model_bias)  #input_size, output_size, bias boolean
    elif args.model_str == "RNN":
        # Initialize the RNN model
        args.model = RNNModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "LSTM":
        # Initialize the LSTM model
        args.model = LSTMModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "GRU":
        args.model = GRUModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "Transformer":
        args.model = TransformerModel(args.input_size, args.output_size)
    else:
        raise NotImplementedError

def k_fold_cross_validation(self, n_splits=5, epochs=10):
    users = self.train_subj_IDs
    user_folds = self.create_user_folds(users, n_splits)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(user_folds):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Set training and validation users
        self.train_users = [users[i] for i in train_idx]
        self.val_users = [users[i] for i in val_idx]
        
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
    time_list = []
    reporter = MemReporter()

    # args.times=1 for now...
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        print(args.model)

        # select algorithm
        server = self.init_algo(args)

        server.train()
        time_list.append(time.time()-start)

    server.plot_results()
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
      
    # Global average
    if args.algorithm != "Local":
        average_data(server.trial_result_path, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
        # Not super sure what "times" is, by default it is 1. Assuming it runs the process multiple times to average out the stochasticity?

    print("All done!")
    reporter.report()
    return server
