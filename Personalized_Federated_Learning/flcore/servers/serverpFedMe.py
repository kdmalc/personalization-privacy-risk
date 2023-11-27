#import os
import copy
#import h5py
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server


class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.beta = args.beta
        self.rs_train_loss_per = []
        self.rs_test_loss_per = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.train()

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        # print("\nBest global accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        print("\nBest loss.")
        print(min(self.rs_test_loss_per))

        self.save_results(personalized=True)
        self.save_global_model()

        self.evaluate(train=False, test=True)
        #if self.num_new_clients > 0:
        #    self.eval_new_clients = True
        #    self.set_new_clients(clientpFedMe)
        #    print(f"\n-------------Fine tuning round-------------")
        #    print("\nEvaluate new clients")
        #    self.evaluate()


    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_loss = []
        for c in self.clients:
            ct, ns = c.test_metrics_personalized()
            tot_loss.append(ct*1.0)
            num_samples.append(ns)
        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, tot_loss

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        tot_loss = []
        for c in self.clients:
            ct, ns = c.train_metrics_personalized()
            tot_loss.append(ct*1.0)
            num_samples.append(ns)

        IDs = [c.ID for c in self.clients]

        return IDs, num_samples, tot_loss

    def evaluate_personalized(self):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_loss = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        
        self.rs_test_loss_per.append(test_loss)
        self.rs_train_loss_per.append(train_loss)

        #self.print_(test_loss, train_loss)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_loss))

