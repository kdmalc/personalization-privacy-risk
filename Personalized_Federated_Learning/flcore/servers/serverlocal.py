# PFLNIID

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


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
        print(f"Outside of loop: self.rs_train_loss TYPE: {type(self.rs_train_loss)}")
        for i in range(self.global_rounds+1):
            print(f"Inside of loop: self.rs_train_loss TYPE: {type(self.rs_train_loss)}")
            ##############################################################
            # I feel like I should be able to delete this... it doesn't affect self.evaluate()...
            #print("Selecting clients")
            #self.selected_clients = self.select_clients()
            ##############################################################

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if i!=0:
                    print("\nEvaluate personalized models")
                    self.evaluate()
                    print(f"Printing rs_train_loss from eval() func on SB")

                    #print(f"len: {len(self.rs_train_loss[-1])}")
                    if type(self.rs_train_loss[-1]) in [int, float]:
                        print(f"rs_train_loss: {self.rs_train_loss[-1]}")
                    else:
                        print(f"len: {len(self.rs_train_loss[-1])}")
                    print()

            print("Selecting clients")
            self.selected_clients = self.select_clients()
            print("CLIENT TRAINING")
            for client in self.selected_clients:
                client.train()
                print(f"Client{client.ID} loss: {client.loss_log[-1]}")

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            print()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        print("\nBest Loss.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # Why is it testing without me telling it to...
        print(min(self.rs_test_loss))

        self.save_results()
        self.save_global_model()