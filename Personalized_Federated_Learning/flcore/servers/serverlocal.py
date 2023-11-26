# PFLNIID

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from datetime import datetime
import os
import numpy as np


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        if self.test_against_all_other_clients:
            # HARD CODE FOR NOW...
            self.clii_on_clij_loss = np.zeros((len(self.clients), len(self.clients), self.global_rounds))
            # Idk if I need this one...
            self.clii_on_clij_numsamples = np.zeros((len(self.clients), len(self.clients), self.global_rounds))

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()


    def average_cross_client_losses(matrix):
        print("AVERAGE_CROSS_CLIENT_LOSSES")
        # Ensure the matrix is square
        assert matrix.shape[0] == matrix.shape[1], "Input matrix must be square"

        # Get the diagonal indices
        diagonal_indices = np.diag_indices(matrix.shape[0])
        print(diagonal_indices)
        # Iterate over each row
        for i in range(matrix.shape[0]):
            # Exclude the diagonal element
            non_diagonal_indices = np.setdiff1d(np.arange(matrix.shape[1]), i)
            print(non_diagonal_indices)
            # Calculate the average excluding the diagonal element
            row_average = np.mean(matrix[i, non_diagonal_indices])
            print(f"row_average.shape: {row_average.shape}, row_average[0:10]: {row_average[0:10]}")
            # Save the average to the diagonal element
            matrix[diagonal_indices[0][i], diagonal_indices[1][i]] = row_average
            print(f"Diagonal indices used: ({diagonal_indices[0][i]}, {diagonal_indices[1][i]})")
            print(f"matrix_diagonal.shape: {matrix[diagonal_indices[0][i], diagonal_indices[1][i]].shape}, matrix_diagonal[:10]: {matrix[diagonal_indices[0][i], diagonal_indices[1][i]][:10]}, matrix[i, i][0:10]: {matrix[i, i, :10]}")
        return matrix


    def train(self):
        self.selected_clients = self.clients
        
        for i in range(self.global_rounds+1):
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if i!=0:
                    #print("\nEvaluate personalized models")
                    self.evaluate(train=self.run_train_metrics)  # I don't understand why train_metrics() is used at all? Need it to log stuff later tho...

                    #print(f"len: {len(self.rs_train_loss[-1])}")
                    #if type(self.rs_train_loss[-1]) in [int, float]:
                    #    print(f"rs_train_loss: {self.rs_train_loss[-1]}")
                    #else:
                    #    print(f"len: {len(self.rs_train_loss[-1])}")
                    print()

            #print("CLIENT TRAINING")
            for client in self.selected_clients:
                # If seq is off then train as normal
                ## If seq is on then only train if client is a live client
                if (self.sequential==False) or ((self.sequential==True) and (client in self.live_clients)):
                    client.train()
                    if self.verbose:
                        print(f"Client {client.ID} loss: {client.loss_log[-1]:0,.3f}")

            # Test current client's model on all other clients
            if self.test_against_all_other_clients:
                for idx_i in range(len(self.selected_clients)):
                    # Select the current element
                    client = self.selected_clients[idx_i]
                    # Iterate through all other elements in the list
                    for idx_j in range(len(self.selected_clients)):
                        # Skip the current element
                        if idx_i != idx_j:
                            other_client = self.selected_clients[idx_j]
                            
                            # Now test the current client's model on the other_client
                            self.clii_on_clij_loss[idx_i, idx_j, i], self.clii_on_clij_numsamples[idx_i, idx_j, i] = other_client.test_metrics(model_obj=client.model)

            #print()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("Breaking")
                break

        # Set the average in the diagonal 
        self.clii_on_clij_loss = self.average_cross_client_losses(self.clii_on_clij_loss)

        self.evaluate(train=False, test=True)
        print("\nBest Loss.")
        print(min(self.rs_test_loss))

        for idx, client in enumerate(self.clients):
            self.cost_func_comps_log.append(client.cost_func_comps_log)
            self.gradient_norm_log.append(client.gradient_norm_log)

        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        str_current_datetime = str(current_datetime)
        self.save_results(save_cost_func_comps=True, save_gradient=True)
        model_path = os.path.join("models", self.dataset, "Local", str(current_datetime))

        trial_result_path = self.result_path + str_current_datetime
        if not os.path.exists(trial_result_path):
            os.makedirs(trial_result_path)

        for client in self.clients:
            client.save_item(client.model, 'local_client_model', item_path=model_path)