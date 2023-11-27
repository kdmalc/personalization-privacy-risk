# Run via: sh <file_name>.sh

nohup python -u main.py -pca_ch 64 -algo APFL

#-lrt = 50
#-al Alpha for APFL, (default: 1)
#-lF=0.0, lD=0.001, lE=0.0001
#-stup = 10
#-normalize_data = True
#-test_split_fraction = 0.2
#-test_split_each_update = False
#-test_split_users = False
#-lm_bias = False --> Probably could change this to true?
