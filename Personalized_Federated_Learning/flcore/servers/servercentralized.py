# PFLNIID

from flcore.clients.clientcentralized import clientCent
from flcore.servers.serverbase import Server
from datetime import datetime
import os


class Centralized(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        base_data_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Data\\Subject_Specific_Files\\'
        client = clientCent(self.args, 
                            ID="_ALL", 
                            dir_path=base_data_path, 
                            condition_number_lst = self.condition_number_lst, 
                            train_slow=False, 
                            send_slow=False)
        self.selected_clients = [client] 
        self.clients.append(client) # This is just technical debt... have to keep it or completely refactor...

        print("Finished creating server and client.")
        
        # self.load_model()


    def train(self):
        #for i in range(self.global_rounds+1):  #Idk why they had +1... maybe their round0 did nothing but init?
        for i in range(self.global_rounds):
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if i!=0:
                    #print("\nEvaluate personalized models")
                    self.evaluate(train=self.run_train_metrics) 

            print("CLIENT TRAINING")
            for client in self.selected_clients:
                client.train()

            #print()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("Breaking")
                break

        self.evaluate(train=False, test=True)
        print("\nBest Loss.")
        print(min(self.rs_test_loss))

        self.cost_func_comps_log.append(client.cost_func_comps_log)
        self.gradient_norm_log.append(client.gradient_norm_log)

        # get current date and time
        current_datetime = datetime.now().strftime("%m-%d_%H-%M")
        # convert datetime obj to string
        str_current_datetime = str(current_datetime)
        self.save_results(save_cost_func_comps=True, save_gradient=True)
        model_path = os.path.join("models", self.dataset, "Local", str(current_datetime))

        #trial_result_path = self.result_path + str_current_datetime
        #if not os.path.exists(trial_result_path):
        #    os.makedirs(trial_result_path)

        for client in self.clients:
            client.save_item(client.model, 'local_client_model', item_path=model_path)