# PFLNIID

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
#from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        print("Serveravg init(): set_slow_clients()")
        self.set_slow_clients()
        print("Serveravg init(): set_clients()")
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            #print("Select clients")
            self.selected_clients = self.select_clients()
            print(f"Selected client IDs: {[client.ID for client in self.selected_clients]}")
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

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                print("DLG think he on the team")
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest loss.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(min(self.rs_test_loss))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        self.eval_new_clients = True
        self.set_new_clients(clientAVG)
        print(f"\n-------------Fine tuning round-------------")
        print("\nEvaluate new clients")
        self.evaluate()