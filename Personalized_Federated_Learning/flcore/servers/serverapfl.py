# PFLNIID

from flcore.clients.clientapfl import clientAPFL
from flcore.servers.serverbase import Server
#from threading import Thread


class APFL(Server):
    def __init__(self, args, times=1):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAPFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()


    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.global_round += 1
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                # If seq is off then train as normal
                ## If seq is on then only train if client is a live client
                if (self.sequential==False) or ((self.sequential==True) and (client in self.live_clients)):
                    client.train()
                    if self.verbose:
                        print(f"Client {client.ID} loss: {client.loss_log[-1]:0,.3f}")

            self.receive_models()
            # I'm not using dlg
            #if self.dlg_eval and i%self.dlg_gap == 0:
            #    self.call_dlg(i)
            self.aggregate_parameters()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_loss], top_cnt=self.top_cnt):
                break

        print("\nBest loss.")
        print(min(self.rs_test_loss))

        self.save_results()

        #####################################################
        # Uhhhh I don't think my dataset can do this... --> It might be able to now? (11/26)
        ## Maybe I should implement the user test split in addition here?
        #self.eval_new_clients = True
        #self.set_new_clients(clientAPFL)
        #print(f"\n-------------Fine tuning round-------------")
        #print("\nEvaluate new clients")
        self.evaluate(train=False)
        #####################################################