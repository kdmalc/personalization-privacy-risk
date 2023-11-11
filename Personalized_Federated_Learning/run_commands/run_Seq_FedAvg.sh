# Run via: sh <file_name>.sh


# FEDAVG WITH LIMITED CLIENT BASE!!
python -u Personalized_Federated_Learning\main.py -lr 1 -pca_ch 64 -algo FedAvg -tr_ids ['METACPHS_S108','METACPHS_S109','METACPHS_S110','METACPHS_S111','METACPHS_S112','METACPHS_S113','METACPHS_S114','METACPHS_S115','METACPHS_S116','METACPHS_S117']


# str(['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']),
python -u main.py -lr 1 -pca_ch 64 -algo FedAvg -static_clients ['METACPHS_S108','METACPHS_S109','METACPHS_S110','METACPHS_S111','METACPHS_S112','METACPHS_S113','METACPHS_S114','METACPHS_S115','METACPHS_S116','METACPHS_S117'],  -live_clients ['METACPHS_S106','METACPHS_S107','METACPHS_S118','METACPHS_S119']
# WITH DIRECTORY:
python -u Personalized_Federated_Learning\main.py -lr 1 -pca_ch 64 -algo FedAvg -seq True -static_clients ['METACPHS_S108','METACPHS_S109','METACPHS_S110','METACPHS_S111','METACPHS_S112','METACPHS_S113','METACPHS_S114','METACPHS_S115','METACPHS_S116','METACPHS_S117'] -live_clients ['METACPHS_S106','METACPHS_S107','METACPHS_S118','METACPHS_S119']

#parser.add_argument('-jr', "--join_ratio", type=float, default=0.3, help="Fraction of clients to be active in training per round")






## SEQUENTIAL TRAINING PARAMS
#parser.add_argument('-seq', "--sequential", type=bool, default=False,
#                    help="Boolean toggle for whether sequential mode is on (for now, mixing current client with previously trained models)")
## default=str(['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']),
#parser.add_argument('-live_clients', "--live_clients", type=str, default='[]',
#                    help="List of current subject ID strings (models will be trained and saved)")
#parser.add_argument('-static_clients', "--static_clients", type=str, default='[]',
#                    help="List of previously trained subject ID strings (models will be uploaded, used in training, but never updated)")
#parser.add_argument('-svlweight', "--static_vs_live_weighting", type=float, default=0.75,
#                    help="Ratio between number of static clients and live clients present in each training round. Set completely arbitrarily for now.")
# ...
## For FedAvg all clients have the same model so code has to change to reflect this
### Current saving regime is broken and saves Local correctly I believe but not FedAvg (no FedAvg directory even...)
#### My default entry is the path with Latest FedAvg filename, despite what the help description says...
#parser.add_argument('-pmd', "--prev_model_directory", type=str, default="C:\\Users\\kdmen\\Desktop\\Research\\personalization-privacy-risk\\Personalized_Federated_Learning\\models\\cphs\\FedAvg_server.pt",
#                    help="Directory name containing all the prev clients models") 
############################################################

#-lrt = 50
#-lF=0.0, lD=0.001, lE=0.0001
#-stup = 10
#-normalize_data = True
#-test_split_fraction = 0.2
#-test_split_each_update = False
#-test_split_users = False
#-lm_bias = False --> Probably could change this to true?

############################################################

#nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedAvg -gr 2000 -did 0 -go dnn > mnist_fedavg.out 2>&1 &
# nohup: This command is used to run a command in the background, and it ensures that the command continues to run even if the terminal session is closed.
#-u: This argument tells Python to run the script in unbuffered mode. It ensures that the output is immediately flushed and displayed, which is useful when redirecting output to files.
#-lbs = batch_size (default: 1202)
#-nc = num_clients (default: 14)
#-jr = join_ratio (default=0.2)
#-nb = num_classes (DOES NOT EXIST FOR US)
#-data
#-m = model (default: LinearRegression)
#-algo = algorithm
#-gr = global_rounds (default: 100, for now)
#-did = device_id (default: 0)
#-go = goal (default: test)
# SHELL COMMANDS
#> mnist_fedavg.out: This part of the command redirects the standard output to a file named mnist_fedavg.out. This means that any text that would have been printed to the terminal by the script will be saved in this file.
#2>&1: This part of the command redirects the standard error (file descriptor 2) to the same location as the standard output (file descriptor 1). It ensures that both standard output and standard error are redirected to the same file.
#&: This symbol at the end of the command runs the entire command in the background, allowing you to continue using the terminal for other tasks.

############################################################

# Idk what the difference between these to is
# python main.py -data mnist -m cnn -algo FedAvg -gr 2500 -did 0 -go cnn # for FedAvg and MNIST