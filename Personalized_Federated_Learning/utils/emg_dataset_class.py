import torch
#import pandas as pd
#import numpy as np
#import pickle

# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

class CustomEMGDataset(torch.utils.data.Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, emg_input, vel_labels, starting_update=10, skip_last_update=True):
        update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        # load data
        ## Passing it in for now so each client doesn't have to reload the dataset
        ### For the full FL implementation, how does this work lol
        
        # Original code from docs
        ########################
        #self.df=pd.read_csv("Stars.csv")
        # extract labels
        #self.df_labels=df[['Type']]
        # drop non numeric columns to make tutorial simpler, in real life do categorical encoding
        #self.df=df.drop(columns=['Type','Color','Spectral_Class'])
        ########################
        
        final_idx = update_ix[-2] if skip_last_update else update_ix[-1]
        # conver to torch dtypes
        self.dataset = torch.tensor(emg_input, dtype=torch.float32)[:final_idx, :]
        self.labels = torch.tensor(vel_labels, dtype=torch.float32)[:final_idx, :]
        
        # Assuming live is False... but that's a whole different refactor lol
        # Idk if I even need this actually
        #self.init_data = self.dataset[starting_update:starting_update+1, :]
        #self.init_labels = self.labels[starting_update:starting_update+1, :]
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        # This doesn't make any sense, why does train point to samples but test points to labels?
        # It appears that I am only using x/y as the inputs for idx anyways...
        if type(idx)==int:
            return self.dataset[idx], self.labels[idx]
        elif (idx.lower()=='x'): # or ('train' in idx.lower()):
            return self.dataset
        elif (idx.lower()=='y'): # or ('test' in idx.lower()):
            return self.labels
        else:
            raise("Not supposed to run")