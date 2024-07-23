# PFLNIID

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
#import os
#from datetime import datetime


class FedAvg(Server):
    def __init__(self, args, times=1):
        super().__init__(args, times=1)
        print("SERVERAVG")

        # select slow clients
        #print("Serveravg init(): set_slow_clients()")
        self.set_slow_clients() # Not using this...
        self.set_clients(clientAVG)
        print(f"Serveravg init(): set_clients() --> set {len(self.clients)} clients")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = [] # Not using this?


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            #print("Select clients")
            self.selected_clients = self.select_clients()
            self.global_round += 1
            #print(f"Selected client IDs: {[client.ID for client in self.selected_clients]}")

            #print("Send models")
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            if self.verbose:
                print("CLIENT TRAINING")
            for client in self.selected_clients:
                # If (seq is off) or (current client is the live seq client) 
                #  then train as normal, else skip training (for dead clients)
                if (self.sequential==False) or ((self.sequential==True) and (client in self.live_clients)):
                    client.train()
                    if self.verbose:
                        print(f"Client {client.ID} loss: {client.loss_log[-1]:0,.3f}")
                # If seq is on but you are a static client, your model shouldn't update, thus no training

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            #if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #    break

        # This ought to be replaced when/if eval_new_clients is integrated
        self.evaluate(train=False, test=True)
        ####################################################
        # Uhhhh I don't think my dataset can do this...
        ## Maybe I should implement the user test split in addition here?
        #self.eval_new_clients = True
        #self.set_new_clients(clientAVG)
        #print(f"\n-------------Fine tuning round-------------")
        #print("\nEvaluate new clients")
        #self.evaluate()
        ####################################################
        
        print("\nBest loss.")
        print(min(self.rs_test_loss))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        for client in self.clients:
            self.cost_func_comps_log.append(client.cost_func_comps_log)
            self.gradient_norm_log.append(client.gradient_norm_log)

        self.save_results(save_cost_func_comps=True, save_gradient=True)