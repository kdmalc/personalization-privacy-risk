# PFLNIID

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from datetime import datetime
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
                client.train()
                if self.verbose:
                    print(f"SL: {client.ID} round {i} loss: {client.loss_log[-1]:0,.5f}")

            #print()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("Breaking")
                break

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