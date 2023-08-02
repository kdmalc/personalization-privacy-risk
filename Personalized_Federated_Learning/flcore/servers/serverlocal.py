# PFLNIID

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
#from threading import Thread
import torch
import os


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()


    def train(self):
        self.selected_clients = self.clients
        with torch.no_grad():
            # subscript global_model with [0] if it is sequential instead of linear model --> does that return just the first layer then?
            self.global_model.weight.fill_(0)
        
        for i in range(self.global_rounds+1):
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if i!=0:
                    print("\nEvaluate personalized models")
                    self.evaluate()  # I don't understand why train_metrics() is used at all? Need it to log stuff later tho...

                    #print(f"len: {len(self.rs_train_loss[-1])}")
                    if type(self.rs_train_loss[-1]) in [int, float]:
                        print(f"rs_train_loss: {self.rs_train_loss[-1]}")
                    else:
                        print(f"len: {len(self.rs_train_loss[-1])}")
                    print()

            #self.selected_clients = self.select_clients()  # FOR LOCAL WE CAN JUST RUN ALL CLIENTS AT ONCE SINCE WE ARE NOT AGGREGATING
            #print(f"Selected client IDs: {[client.ID for client in self.selected_clients]}")
            #print("CLIENT TRAINING")
            for client in self.selected_clients:
                client.train()
                print(f"Client{client.ID} round {i} loss: {client.loss_log[-1]:0,.3f}")

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            print()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("Breaking")
                break

        self.evaluate(train=False, test=True)
        print("\nBest Loss.")
        print(min(self.rs_test_loss))

        # So how do I do this given that not all clients will have trained and thus some of these will be empty...
        # No this is post training so they should all be filled, just different lengths...
        # OHH my problem is that since I am only running for 5 training rounds some clients haven't trained at all lol
        for idx, client in enumerate(self.clients):
            #self.cost_func_comps_dict[idx] = client.cost_func_comps_log
            #self.gradient_dict[idx] = client.gradient_norm_log
            self.cost_func_comps_log.append(client.cost_func_comps_log)
            self.gradient_norm_log.append(client.gradient_norm_log)

        self.save_results(save_cost_func_comps=True, save_gradient=True)
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, "Local")
        for client in self.clients:
            client.save_item(client.model, 'local_client_model', item_path=model_path)
        # No idea where this global model is coming from? Why did they save it...
        self.save_global_model()