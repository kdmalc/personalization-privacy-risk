# PFLNIID

import numpy as np
import os
import torch


def read_data(dataset, idx, is_train=True):
    # KAI: THIS NEEDS TO BE UPDATED, EITHER ACTUALLY PUT DATA THERE OR COMPLETELY REPLACE THIS
    # IDK WHICH
    # What's the difference between this and the Dataset class I made?...
    # This is called within read_client_data...
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        #X_train = torch.Tensor(train_data['x']).type(torch.float32)
        #y_train = torch.Tensor(train_data['y']).type(torch.int64)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.float32)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        #X_test = torch.Tensor(test_data['x']).type(torch.float32)
        #y_test = torch.Tensor(test_data['y']).type(torch.int64)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.float32)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data