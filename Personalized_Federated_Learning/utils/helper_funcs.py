import ast
import os
import h5py

def convert_cmd_line_str_lst_to_type_lst(list_as_string, datatype, verbose=False):
    '''
    Safely convert the command line entered list (which Python uses a string representation of) to a list of the correct datatype
    Inputs:
        datatype: int or str, depending on what you're doing
    '''
    #try:
    # Using ast.literal_eval to safely evaluate the string
    if type(list_as_string)!=str:
        #print("CONVERTING TO STRING")
        list_as_string = str(list_as_string)
    #print(f"list_as_string")
    #print(list_as_string)
    temp_lst = ast.literal_eval(list_as_string)
    if datatype==int:
        if not isinstance(temp_lst, list) or not all(isinstance(item, datatype) for item in temp_lst):
            raise ValueError(f"Invalid input. Must be a list of {datatype}.")
    elif datatype==str:
        # Initialize an empty list to store the string elements
        string_list = []

        # Iterate through the characters and add them to the result list as strings
        current_string = ""
        for char in temp_lst:
            #print(f"char: {char}")
            #print(f"current_string: {current_string}")
            #print(f"string_list: {string_list}")
            #print()
            
            if char=='[' or char==']' or char==',' or char==' ':
                #print("MATCH")
                pass
            elif char == "'":
                #print("QUOTATION")
                # When encountering a single quote, it indicates the start or end of a string
                if current_string:
                    string_list.append(current_string)
                    current_string = ""
            else:
                #print("ELSE")
                current_string += char
    else:
        raise ValueError(f"{datatype} is not a supported datatype, please enter <str> or <int> (note: no quotation marks)")
    #except (ValueError, SyntaxError):
    #    print(f"Invalid input. Please provide a valid list of {datatype}.")
    if verbose:
        print(list_as_string)
        print(temp_lst)
    return temp_lst

import numpy as np

def average_kfold_data(algorithm="", dataset="", goal="", n_folds=5):
    train_losses, test_losses = get_all_results_for_one_algo(algorithm, dataset, goal, n_folds)
    
    # Calculate best accuracy for each fold
    best_train_acc = []
    best_test_acc = []
    for i in range(n_folds):
        best_train_acc.append(train_losses[i].max())
        best_test_acc.append(test_losses[i].max())
    
    print("Train accuracy:")
    print("std for best accuracy:", np.std(best_train_acc))
    print("mean for best accuracy:", np.mean(best_train_acc))
    
    print("\nTest accuracy:")
    print("std for best accuracy:", np.std(best_test_acc))
    print("mean for best accuracy:", np.mean(best_test_acc))
    
    return train_losses, test_losses

def get_all_results_for_one_algo(algorithm="", dataset="", goal="", n_folds=5):
    train_losses = []
    test_losses = []
    for i in range(n_folds):
        # Verify that these are saved somewhere (still need to implement this somewhere...)
        file_name = f"{dataset}_{algorithm}_{goal}_fold{i}"
        fold_data = np.array(read_data_then_delete(file_name, delete=False))
        train_losses.append(fold_data[:, 0])  # Assuming first column is train loss
        test_losses.append(fold_data[:, 1])   # Assuming second column is test loss
    return train_losses, test_losses

def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    # I think this needs to be fixed... I don't know how to get the hf file to work without testing it...
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc