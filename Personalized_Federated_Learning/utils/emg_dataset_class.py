import torch
from math import ceil
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
        

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, sequence_length):
        self.samples = torch.Tensor(samples)
        self.labels = torch.Tensor(labels)
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        # Assuming each sample is a sequence of length sequence_length
        #return self.samples[index:index+self.sequence_length, :], self.labels[index:index+self.sequence_length, :]
        start_idx = index*self.sequence_length
        end_idx = (index+1)*self.sequence_length
        return self.samples[start_idx:end_idx, :], self.labels[start_idx:end_idx, :]

    def __len__(self):
        #return self.samples.shape[0] - self.sequence_length + 1
        return (self.samples.shape[0] // self.sequence_length) + 1
        

class EMG3DDataset(torch.utils.data.Dataset):
    def __init__(self, emg_input, vel_labels, input_size, output_size, sequence_length):
        emg_input = torch.tensor(emg_input)
        vel_labels = torch.tensor(vel_labels)
        print(f"emg_input.shape: {emg_input.shape}")
        print(f"vel_labels.shape: {vel_labels.shape}")

        # Calculate inferred dimensions based on sequence length and sizes
        #inferred_dimension_input = emg_input.numel() // sequence_length // input_size
        #inferred_dimension_labels = vel_labels.numel() // sequence_length // output_size
        #inferred_dimension_input = int(ceil(emg_input.numel() / sequence_length / input_size))
        #inferred_dimension_labels = int(ceil(vel_labels.numel() / sequence_length / output_size))
        #assert()
        #self.reshaped_data = emg_input.reshape(inferred_dimension_input, sequence_length, input_size)
        #self.reshaped_targets = vel_labels.reshape(inferred_dimension_labels, sequence_length, output_size)


        #################################################################################################
        # Attempt at padding...
        ## Ran into some memory issue ...
        ### [enforce fail at C:\b\abs_f0dma8qm3d\croot\pytorch_1669187301762\work\c10\core\impl\alloc_cpu.cpp:81] data. DefaultCPUAllocator: not enough memory: you tried to allocate 62114595840 bytes.
        #import torch.nn.functional as F
        ## Calculate inferred dimensions based on sequence length and sizes
        #inferred_dimension_input = emg_input.numel() // sequence_length // input_size
        #inferred_dimension_labels = vel_labels.numel() // sequence_length // output_size
        ## Calculate the number of elements needed for padding
        #padding_elements_input = (sequence_length * input_size) - (emg_input.numel() % (sequence_length * input_size))
        #padding_elements_labels = (sequence_length * output_size) - (vel_labels.numel() % (sequence_length * output_size))
        ## Pad the input tensor
        #emg_input = F.pad(emg_input, (0, padding_elements_input))
        ## Pad the labels tensor
        #vel_labels = F.pad(vel_labels, (0, padding_elements_labels))
        ## Reshape the data
        #self.reshaped_data = emg_input.reshape(inferred_dimension_input, sequence_length, input_size)
        #################################################################################################


        # If padding doesn't work I could also probably just calculate two indices and max/min them with the bounds
        ## May get batches that are different sizes but idk if the model will care/know?
        # 1: Sample a subset of the input/labels
        # 2: Ensure sizes work out, adjust subset idxs as needed
        # 3: Reshape the subset
        #self.reshaped_data = emg_input.reshape(inferred_dimension_input, sequence_length, input_size)
        #self.reshaped_targets = vel_labels.reshape(inferred_dimension_labels, sequence_length, output_size)

        ######################################################################################################
        # Your original tensor
        # Calculate the original and target number of elements
        num_sample_elements = emg_input.numel()
        num_label_elements = vel_labels.numel()
        # Your rounded integer dimensions for the reshaped tensor
        inferred_dimension_input = int(ceil(emg_input.numel() / sequence_length / input_size))
        inferred_dimension_labels = int(ceil(vel_labels.numel() / sequence_length / output_size))
        # Your rounded integer dimensions for the reshaped tensor
        new_sample_dimensions = (inferred_dimension_input, sequence_length, input_size)
        new_label_dimensions = (inferred_dimension_labels, sequence_length, output_size)
        # Calculate the target number of elements in the reshaped tensor
        target_sample_elements = torch.tensor(new_sample_dimensions).prod().item()
        target_label_elements = torch.tensor(new_label_dimensions).prod().item()
        # SAMPLES: Check if the reshaping is possible
        if num_sample_elements != target_sample_elements:
            # Adjust dimensions to make reshaping possible
            adjustment_factor = num_sample_elements // sequence_length // input_size
            adjusted_dimensions = (adjustment_factor, sequence_length, input_size)
            # Check if the adjusted dimensions are still not enough
            while torch.tensor(adjusted_dimensions).prod().item() < num_sample_elements:
                adjustment_factor += 1
                adjusted_dimensions = (adjustment_factor, sequence_length, input_size)
            new_sample_dimensions = adjusted_dimensions
        self.reshaped_data = emg_input.reshape(*new_sample_dimensions)
        # LABELS: Check if the reshaping is possible
        if num_label_elements != target_label_elements:
            # Adjust dimensions to make reshaping possible
            adjustment_factor = num_label_elements // sequence_length // output_size
            adjusted_dimensions = (adjustment_factor, sequence_length, output_size)
            # Check if the adjusted dimensions are still not enough
            while torch.tensor(adjusted_dimensions).prod().item() < num_label_elements:
                adjustment_factor += 1
                adjusted_dimensions = (adjustment_factor, sequence_length, output_size)
            new_label_dimensions = adjusted_dimensions
        self.reshaped_labels = vel_labels.reshape(*new_sample_dimensions)

    def __len__(self):
        # I think this one is the batching dimension which is what it should care about?...
        return self.reshaped_data.shape[0]

    def __getitem__(self, idx):
        # Lazy way: Let Trainloader deal with it
        #return self.emg_input[idx, :], self.vel_labels[idx, :]
        # Error: for some reason, at some point idx='x' (from trainloader?) and this breaks it...
        return self.reshaped_data[idx, :, :], self.reshaped_targets[idx, :, :]
    

class EMG3DDatasetBATCHED(torch.utils.data.Dataset):
    '''I don't think I need to be doing all this stuff with the batching, in the Dataset obj...'''
    def __init__(self, emg_input, vel_labels, batch_size, input_size, output_size, sequence_length):
        self.emg_input = emg_input
        self.vel_labels = vel_labels
        print(f"emg_input.shape: {emg_input.shape}")
        print(f"vel_labels.shape: {vel_labels.shape}")

        self.input_size = input_size
        self.output_size = output_size

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.original_sequence_length = self.emg_input.shape[0]
        self.original_sequence_length_labels = self.vel_labels.shape[0]
        assert(self.emg_input.shape[0] == self.vel_labels.shape[0])
        self.total_samples = self.emg_input.size
        self.total_labels = self.vel_labels.size

        # Idk what to do if this isn't an int...
        ## PyTorch must handle this somehow, somewhere... probably in trainloader..........
        ## Can I just be using a regular Dataset lol
        self.num_sample_batches = self.emg_input.size / self.batch_size / self.sequence_length / self.input_size
        self.num_label_batches = self.vel_labels.size / self.batch_size / self.sequence_length / self.input_size

        self.batched_data = self.emg_input.reshape(self.num_sample_batches, self.batch_size, self.sequence_length, self.input_size)
        self.batched_targets = self.vel_labels.reshape(self.num_label_batches, self.batch_size, self.sequence_length, self.output_size)

    def __len__(self):
        return self.original_sequence_length

    def __getitem__(self, idx):
        return self.batched_data[idx], self.batched_targets[idx]


class ARCHIVED_EMG3DDataset(torch.utils.data.Dataset):
    def __init__(self, emg_input, vel_labels, batch_size, input_size, output_size, sequence_length):
        self.emg_input = emg_input
        self.vel_labels = vel_labels
        print(f"emg_input.shape: {emg_input.shape}")
        print(f"vel_labels.shape: {vel_labels.shape}")

        self.input_size = input_size
        self.output_size = output_size

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.original_sequence_length = self.emg_input.shape[0]
        self.original_sequence_length_labels = self.vel_labels.shape[0]
        assert(self.emg_input.shape[0] == self.vel_labels.shape[0])
        self.total_samples = self.emg_input.size
        self.total_labels = self.vel_labels.size

    def __len__(self):
        return self.original_sequence_length

    def __getitem__(self, idx):
        #start_idx = idx * self.batch_size * self.sequence_length
        #end_idx = (idx + 1) * self.batch_size * self.sequence_length
        #batch_data = self.data[:, start_idx:end_idx].reshape(self.batch_size, self.sequence_length, -1)
        ## Assuming you have target data, adjust the target retrieval accordingly
        #batch_targets = self.targets[:, start_idx:end_idx].reshape(self.batch_size, self.sequence_length, -1)
        #return batch_data, batch_targets
    
        #start_idx = idx * self.batch_size * self.sequence_length
        #end_idx = (idx + 1) * self.batch_size * self.sequence_length
        start_idx = idx * (self.original_sequence_length // self.batch_size)
        end_idx = (idx + 1) * (self.original_sequence_length // self.batch_size)
        # Ensure the end index doesn't exceed the total number of samples
        end_idx = min(end_idx, self.original_sequence_length)
        # Make the code robust for when there is a remainder on the division
        remaining_samples = (end_idx - start_idx) % self.batch_size
        #print(f"REMAINING SAMPLES: {remaining_samples}")
        # Drop some from the start
        adjusted_start_idx = start_idx + remaining_samples
        # Or could drop some from the end...
        #adjusted_end_idx = end_idx - remaining_samples
        #assert(end_idx - adjusted_start_idx > 0)
        if end_idx - adjusted_start_idx < 0:
            print("AVAILABLE BATCH WAS SMALLER THAN 0... (EMG3DDataset)")
            # Just fill up the final batch... or should I skip it...
            adjusted_start_idx -= 2*remaining_samples

        # In the code it returns a 4D tensor... 64x64x55x64...
        batch_data = self.emg_input[adjusted_start_idx:end_idx, :].reshape(self.batch_size, -1, self.input_size)
        batch_targets = self.vel_labels[adjusted_start_idx:end_idx, :].reshape(self.batch_size, -1, self.output_size)
        return batch_data, batch_targets