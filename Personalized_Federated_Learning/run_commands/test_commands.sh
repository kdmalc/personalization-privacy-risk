# This is to test my theory about local vs FedAvg, training over all the data for just 1 client. 
# Let's see the differences between Local and FedAvg, this may tell us functional info about the stat hetero
python -u main.py -tr_ids ['METACPHS_S106'] -con_num [1,2,3,4,5,6,7,8] -gr 100 -lr 1 -lrt 25 -lD 0.00001 -lE 0.000001 -algo FedAvg
python -u main.py -tr_ids ['METACPHS_S106'] -con_num [1,2,3,4,5,6,7,8] -gr 100 -lr 1 -lrt 25 -lD 0.00001 -lE 0.000001 -algo Local

# -tr_ids = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 
#    'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 
#    'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 
#    'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']
# -con_num = [1]
#-lrt = 50
#-lF=0.0, lD=0.001, lE=0.0001
#-stup = 10
#-normalize_data = True
#-test_split_fraction = 0.2
#-test_split_each_update = False
#-test_split_users = False
#-lm_bias = True
#-pca_ch = 10
#-bs = 1202 (AKA full update)
#-localepochs = 3