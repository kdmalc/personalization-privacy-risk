# Run via: sh <file_name>.sh

nohup python -u main.py -lr 1 -pca_ch 64 -algo FedAvg

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