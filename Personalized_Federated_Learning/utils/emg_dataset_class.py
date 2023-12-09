import torch
from math import ceil

# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

class CustomEMGDataset(torch.utils.data.Dataset):
    '''This needs to be refactored...
    1. To reflect the test/train splits natively (proabbly upstream of this...)
    2. If batches are being fed in, it's not clear what update_ix is even doing here (other than apparently test split stuff...)
    3. Moved starting_update to CB _load_train_data()
    4. If I could remove sequence length in my other Dataset, I could combine these into one...
    5. How would I make this real time? It doesn't really know right... it seems like this code is update-batches-ready (except test split issues)'''

    def __init__(self, emg_input, vel_labels, skip_last_update=True):
        update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        
        # This really ought to be built into the testing case...
        final_idx = update_ix[-2] if skip_last_update else update_ix[-1]
        # conver to torch dtypes
        self.dataset = torch.tensor(emg_input, dtype=torch.float32)[:final_idx, :]
        self.labels = torch.tensor(vel_labels, dtype=torch.float32)[:final_idx, :]
    
    def __len__(self):
        # Which dimension is this lol
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # I'm not sure what causes the 'x' and 'y' indexing issue but I think this shouldn't be happening
        ## Eg if it is then I need to fix it
        ### DEPRECIATED CODE:
        # It appears that I am only using x/y as the inputs for idx anyways...
        #if type(idx)==int:
        #    return self.dataset[idx], self.labels[idx]
        #elif (idx.lower()=='x'): # or ('train' in idx.lower()):
        #    return self.dataset
        #elif (idx.lower()=='y'): # or ('test' in idx.lower()):
        #    return self.labels
        #else:
        #    raise("Not supposed to run")
        
        return self.dataset[idx], self.labels[idx]
        

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, sequence_length):
        self.samples = torch.Tensor(samples)
        self.labels = torch.Tensor(labels)
        self.sequence_length = sequence_length

    def __len__(self):
        #return self.samples.shape[0] - self.sequence_length + 1
        return (self.samples.shape[0] // self.sequence_length) + 1

    def __getitem__(self, index):
        # Assuming each sample is a sequence of length sequence_length
        #return self.samples[index:index+self.sequence_length, :], self.labels[index:index+self.sequence_length, :]
        start_idx = index*self.sequence_length
        end_idx = (index+1)*self.sequence_length
        return self.samples[start_idx:end_idx, :], self.labels[start_idx:end_idx, :]
        