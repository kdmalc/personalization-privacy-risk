keys = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']

num_conds = 8

num_channels = 64

# Default number of k-folds
cv = 5 # Changed to 5 from 10 because the smallest class in cross val only has (had?) 7 instances

my_metrics_cols=['Algorithm', 'One Off Acc', 'CV Acc', 'K Folds']