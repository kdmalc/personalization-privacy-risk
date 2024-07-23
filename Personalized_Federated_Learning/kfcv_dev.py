
import torch
from sklearn.model_selection import KFold
import numpy as np
import time

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

#def create_unified_fold_test_dataloader(user_datasets, user_labels, batch_size=32):
#    datasets = [UserTimeSeriesDataset(user_data, user_label) 
#                for user_data, user_label in zip(user_datasets, user_labels)]
#    combined_dataset = ConcatDataset(datasets)
#    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
#    return dataloader
# Example usage
#user_datasets = [np.random.randn(20770, 64) for _ in range(5)]  # 5 users, each with [20770, 64] data
#user_labels = [np.random.randn(20770, 2) for _ in range(5)]  # 5 users, each with [20770, 2] labels
#test_dataloader = create_crossval_test_dataloader(user_datasets, user_labels)