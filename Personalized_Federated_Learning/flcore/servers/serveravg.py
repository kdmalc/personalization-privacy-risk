# PFLNIID

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import os
from datetime import datetime


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        #print("Serveravg init(): set_slow_clients()")
        self.set_slow_clients()
        self.set_clients(clientAVG)
        print(f"Serveravg init(): set_clients() --> set {len(self.clients)} clients")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            #print("Select clients")
            self.selected_clients = self.select_clients()
            #print(f"Selected client IDs: {[client.ID for client in self.selected_clients]}")
            #print("Send models")
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            print("CLIENT TRAINING")
            for client in self.selected_clients:
                client.train()
                print(f"Client{client.ID} loss: {client.loss_log[-1]:0,.3f}")

            self.receive_models()
            # I'm not using dlg
            #if self.dlg_eval and i%self.dlg_gap == 0:
            #    print("DLG think he on the team")
            #    self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # This ought to be replaced when/if eval_new_clients is integrated
        self.evaluate(train=False, test=True)
        
        print("\nBest loss.")
        print(min(self.rs_test_loss))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # Kai's added code for logging
        for idx, client in enumerate(self.clients):
            self.cost_func_comps_log.append(client.cost_func_comps_log)
            self.gradient_norm_log.append(client.gradient_norm_log)
        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        str_current_datetime = str(current_datetime)
        self.save_results(save_cost_func_comps=True, save_gradient=True)
        model_path = os.path.join("models", self.dataset, "Local", str(current_datetime))
        #
        trial_result_path = self.result_path + str_current_datetime
        if not os.path.exists(trial_result_path):
            os.makedirs(trial_result_path)
        #
        for client in self.clients:
            client.save_item(client.model, 'local_client_model', item_path=model_path)
        # No idea where this global model is coming from? Why did they save it...
        self.save_global_model()

        ## The original code
        #self.save_results()
        #self.save_global_model()

        ####################################################
        # Uhhhh I don't think my dataset can do this...
        ## Maybe I should implement the user test split in addition here?
        #self.eval_new_clients = True
        #self.set_new_clients(clientAVG)
        #print(f"\n-------------Fine tuning round-------------")
        #print("\nEvaluate new clients")
        #self.evaluate()
        ####################################################