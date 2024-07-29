import copy
#import torch
#import numpy as np
from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
#from threading import Thread
import time


class PerAvg(Server):
    def __init__(self, args, times=1):
        super().__init__(args, times)

        self.beta = args.beta
        if self.batch_size>(args.update_batch_length/2):
            raise ValueError(f"For PerFedAvg, batch_size {self.batch_size} must be less than half of the update length {args.update_batch_length} (due to two step optimization).")

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPerAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = [] #Idk what this is, its from PFLib

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.global_round += 1
            # send all parameter for clients
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                #self.evaluate_one_step()
                self.evaluate()  # <-- This is what it is in serveravg...

            # choose several clients to send back updated model to server
            for client in self.selected_clients:
                # If seq is off then train as normal
                ## If seq is on then only train if client is a live client
                if (self.sequential==False) or ((self.sequential==True) and (client in self.live_clients)):
                    client.train()
                    if self.verbose:
                        print(f"Client {client.ID} loss: {client.loss_log[-1]:0,.3f}")
                    # Why is this here twice... (it appeared twice in their codebase, idk)
                    client.train()

            self.receive_models()
            #if self.dlg_eval and i%self.dlg_gap == 0:
            #    self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            #if self.auto_break and self.check_done(acc_lss=[self.rs_test_loss], top_cnt=self.top_cnt):
            #    break

        #self.evaluate(train=False, test=True)  # Is this wrong? Causing the test spike jump?
            
        #if self.num_new_clients > 0:
        #    self.eval_new_clients = True
        #    self.set_new_clients(clientPerAvg)
        #    print(f"\n-------------Fine tuning round-------------")
        #    print("\nEvaluate new clients")
        #    self.evaluate()

        print("\nBest loss.")
        print(min(self.rs_test_loss))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # This is run in severavg, I'm assuming it must need to be run here
        ## Am further assuming that it won't break due to seq...
        for client in self.clients:
            self.cost_func_comps_log.append(client.cost_func_comps_log)
            self.gradient_norm_log.append(client.gradient_norm_log)

        self.save_results(save_cost_func_comps=True, save_gradient=True)


    def evaluate_one_step(self, acc=None, loss=None):
        models_temp = []
        # Only use selected clients since real trials won't have past clients?...
        ## I won't run this py file for the lab tho... leave it as self.clients
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()
        stats = self.test_metrics()
        # set the local model back on clients for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)
        stats_train = self.train_metrics()
        # set the local model back on clients for training process
        #for i, c in enumerate(self.clients):
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)
        #accs = [a / n for a, n in zip(stats[2], stats[1])]
        test_loss = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        
        if acc == None:
            self.rs_test_loss.append(test_loss)
            if self.sequential:
                # seq_stats <-- [curr_live_loss, curr_live_num_samples, curr_live_IDs, prev_live_loss, prev_live_num_samples, prev_live_IDs, unseen_live_loss, unseen_live_num_samples, unseen_live_IDs]
                # Hmm do I need to save/use the actual IDs at all? Do I care? Don't think so...
                seq_stats = stats[3]
                if len(seq_stats[0])!=0:
                    self.curr_live_rs_test_loss.append(sum(seq_stats[0])/sum(seq_stats[1]))
                if len(seq_stats[3])!=0:
                    self.prev_live_rs_test_loss.append(sum(seq_stats[3])/sum(seq_stats[4]))
                if len(seq_stats[6])!=0:
                    self.unseen_live_rs_test_loss.append(sum(seq_stats[6])/sum(seq_stats[7]))
        else:
            acc.append(test_loss)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # self.print_(test_acc, train_acc, train_loss)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        #print("Averaged Test Loss: {:.4f}".format(test_loss))
        print(f"Averaged Test Loss: {test_loss}")
        #print("Std Test Accuracy: {:.4f}".format(np.std(accs)))