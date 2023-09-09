nohup python -u main.py -gr 200 -lr 0.01 -lrt 25 -pca_ch 10 -lD 0.00001 -lE 0.000001 -lm_bias True -algo Local

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