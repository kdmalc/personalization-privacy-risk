

echo "lD=0.000001, lE=0.0000001"
echo "lD=0.01, lE=0.000001"
echo "lD=0.0001, lE=0.01"

echo "lD=0.01, lE=0.000001"
echo "lD=0.00001, lE=0.000001"



echo "lD=1, lE=1"
echo "lD=10, lE=1"
echo "lD=1, lE=10"
echo "lD=0.05, lE=0.01"
echo "lD=0.05, lE=0.001"
echo "lD=0.0001, lE=0.01"







echo "Run 1:"
nohup python -u main.py -gr 200 -lr 1 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 1 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local
echo "Run 2:"
nohup python -u main.py -gr 200 -lr 0.0001 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 0.0001 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local
echo "Run 3:"
nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local




## SEPT 5TH MANUAL GRID SEARCH
# These have been the working params so far
'''
echo "lD=0.001, lE=0.0001"
echo "Run 1:"
nohup python -u main.py -gr 200 -lr 1 -lrt 25 -pca_ch 10 -lD 0.001 -lE 0.0001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 1 -lrt 25 -pca_ch 10 -lD 0.001 -lE 0.0001 -lm_bias True -algo Local
echo "Run 2:"Alpha_n = n_inf / Tau_n
nohup python -u main.py -gr 200 -lr 0.0001 -lrt 25 -pca_ch 10 -lD 0.001 -lE 0.0001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 0.0001 -lrt 25 -pca_ch 10 -lD 0.001 -lE 0.0001 -lm_bias True -algo Local
echo "Run 3:"
nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.001 -lE 0.0001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.001 -lE 0.0001 -lm_bias True -algo Local

# These were the original values used in 599 (minus lF being 0 now)
echo "lD=0.00001, lE=0.000001"
echo "Run 1:"
nohup python -u main.py -gr 200 -lr 1 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 1 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local
echo "Run 2:"
nohup python -u main.py -gr 200 -lr 0.0001 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 0.0001 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local
echo "Run 3:"
nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo FedAvg
nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local
'''

#-lrt = 50
#-lF=0.0, lD=0.001, lE=0.0001
#-stup = 10
#-normalize_data = True
#-test_split_fraction = 0.2
#-test_split_each_update = False
#-test_split_users = False
#-lm_bias = False --> Probably could change this to true?
#-bs = 1202 (AKA full update)
#-localepochs = 3