import h5py
import numpy as np
import os


def average_data(base_server_file_path, algorithm="", dataset="", goal="", times=10):
    #test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
    test_loss = get_all_results_for_one_algo(base_server_file_path, algorithm, dataset, goal, times)

    #max_accurancy = []
    min_loss = []
    for i in range(times):
        #max_accurancy.append(test_acc[i].max())
        min_loss.append(test_loss[i].min())

    #print("std for best accurancy:", np.std(max_accurancy))
    #print("mean for best accurancy:", np.mean(max_accurancy))
    print("std for best loss:", np.std(min_loss))
    print("mean for best loss:", np.mean(min_loss))


def get_all_results_for_one_algo(base_server_file_path, algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal# + "_" + str(i)
        full_file_name = os.path.join(base_server_file_path, file_name)
        test_acc.append(np.array(read_data_then_delete(full_file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    #file_path = "../results/" + file_name + ".h5"
    file_path = file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_loss = np.array(hf.get('rs_test_loss'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_loss))

    return rs_test_loss