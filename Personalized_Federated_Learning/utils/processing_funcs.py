import torch

def normalize_2D_tensor(s_temp):
    '''Normalizes a 2D tensor. Goal is to have inputs on the range 0-1, NOT a norm of 1.'''

    # Compute min and max values
    # I think my cols and rows are actually flipped but it doesn't matter...
    min_values_rows, _ = torch.min(s_temp, dim=0)
    min_values_cols, _ = torch.min(s_temp, dim=1)
    min_val = torch.min(torch.min(min_values_rows), torch.min(min_values_cols))

    max_values_rows, _ = torch.max(s_temp, dim=0)
    max_values_cols, _ = torch.max(s_temp, dim=1)
    max_val = torch.max(torch.max(max_values_rows), torch.max(max_values_cols))

    return (s_temp - min_val) / (max_val - min_val)
 

def normalize_tensor(input_tensor):
    '''Normalizes a tensor of any dimensions. Goal is to have inputs on the range 0-1, NOT a norm of 1.'''
    
    # Compute min and max values across all dimensions
    min_values = torch.min(input_tensor)
    max_values = torch.max(input_tensor)

    return (input_tensor - min_values) / (max_values - min_values)