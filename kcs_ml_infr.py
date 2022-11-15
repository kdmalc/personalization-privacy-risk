import numpy as np
import time
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model

random.seed(a=1)

def fit_ml_algo(algo, X_train, y_train, cv, verbose=False, num_decimals=3, testing=False):
    '''Runs given algorithm and returns the accuracy metrics'''
    
    model = algo.fit(X_train, y_train)
    
    # Notice that this is tested on the data it just trained on, so this is a bad accuracy metric to use
    acc = round(model.score(X_train, y_train) * 100, 3)
    
    # Cross Validation - this fixes that issue of validating on the data that the model was trained on
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs=-1)
    # Cross-validation metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, num_decimals)
    #pre_cv = round(metrics.precision_score(y_train, train_pred) * 100, num_decimals)
    #rec_cv = round(metrics.recall_score(y_train, train_pred) * 100, num_decimals)
    
    if verbose:
        print("Training predictions:")
        print(train_pred)
        print("Ground Truth:")
        print(y_train)
        print(f"One Off Accuracy: {acc}")
        print(f"CV Accuracy: {acc_cv}")
    
    if testing:
        return train_pred, acc, acc_cv, model
    
    return train_pred, acc, acc_cv
    
    
def train_test_val_split(input_df, label_df, rng_seed=2, validation=False, test_percent=0.3, val_percent=0.3):
    '''
    I don't think I need a validation set if I'm just doing cross_validation since that should take care of it for me
    '''
    
    x_train = input_df.copy(deep=True)
    y_train_reg = label_df

    ## TRAIN / TEST
    # Stratify might be good to ensure that all classes are represented, I'm not sure if it'll do that by default
    X_train, X_test, y_train, y_test = train_test_split(
        x_train, y_train_reg, test_size=test_percent, random_state=rng_seed, shuffle=True)
    # Not sure how shuffle and random state interact...

    # Should just use cross val instead of manually making val split
    if validation:
        ## TRAIN / VAL
        X_train_pv, X_val, y_train_pv, y_val = train_test_split(
            X_train, y_train, test_size=val_percent, random_state=rng_seed, shuffle=True)

        return X_train_pv, y_train_pv, X_test, y_test, X_val, y_val
    else:
        return X_train, y_train, X_test, y_test, 0, 0
        

def train_model(my_model, X_train, y_train, cv, res_df, verbose=False, dec_num='NA'):
    ''''''
    
    train_pred_log, acc, acc_cv = fit_ml_algo(my_model, X_train, y_train, cv)
    if verbose:
        print(f"{str(my_model)}")
        print(f"Accuracy: {acc}")
        print(f"Accuracy CV 10-Fold: {acc_cv}")
        print()

    my_metrics_cols = ['Algorithm', 'One Off Acc', 'CV Acc', 'K Folds', 'N']
    temp_df = pd.DataFrame([[str(my_model), acc, acc_cv, cv, dec_num]], columns=my_metrics_cols)
    res_df = pd.concat((res_df, temp_df))
    
    return res_df
    

def test_model(model_name, X_train, y_train, X_test, y_test, test_df, cv, num_decimals=3, verbose=False, my_cols=['Algorithm', 'CV Acc', 'Test Acc', 'K Folds', 'N'], dec_num='NA'):
    ''''''
    
    _, _, acc_cv, trained_model = fit_ml_algo(model_name, X_train, y_train, cv, testing=True)
    y_pred = trained_model.predict(X_test)
    test_acc = round(metrics.accuracy_score(y_test, y_pred) * 100, num_decimals)
    
    if verbose:
        print(str(model_name))
        print(f"CV Accuracy: {acc_cv}")
        print(f"Test Accuracy: {test_acc}")
        print()
        
    temp_df = pd.DataFrame([str(model_name), acc_cv, test_acc, cv, dec_num], index=my_cols).T
    test_df = pd.concat((test_df, temp_df))
    return test_df