# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:40:58 2021

@author: joa24jm

This file reads in the prepared dataframe for machine learning
and tries to predict the tinnitus occurence for the ema daily questionnaire.
It contains 2179 users from ten countries with data collected between 2014-04 and 2021-02.


"""
p_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'

#%% imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import confusion_matrix, mean_absolute_error
import datetime
from pathlib import Path
import numpy as np
from utilities import regression_report

#%% read in df
base_path = Path(__file__).parent.parent.parent
file_path = (base_path / 'data/03_processed/df_equal_splits_with_age_with_question2_question_3.csv').resolve()
df = pd.read_csv(file_path)

#%% select features and target
features = ['AT', 'CA', 'CH','DE','GB', 'IT', 'NL', 'NO', 'RU', 'US', # countries
            'autumn', 'spring', 'summer', 'winter',                  # season
            'Male', 'age',                                 # demographics
            'question4', 'question5', 'question6', 'question7'
            ]      # EMAs

X = df[features] # all columns except for the last
y = df['question2']  # How loud is the tinnitus right now? -> Regression problem


# split up data into train and test, stratify on y, set random_state and shuffle
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42,
                                                    shuffle = True)

# save to csv for further analysis
x_train.to_csv(p_loc + 'data/04_models/x_train.csv', index_label = 'index')
y_train.to_csv(p_loc + 'data/04_models/y_train.csv', index_label = 'index')
x_test.to_csv(p_loc + 'data/04_models/x_test.csv', index_label = 'index')
y_test.to_csv(p_loc + 'data/04_models/y_test.csv', index_label = 'index')



#%% prepare gridsearch for both random forest (rf) and gradient boosting classifier (gb)
# list of default classifiers

# safe date of approach
tday = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')

clfs = [GradientBoostingRegressor()]

params_gb = {'learning_rate': [0.1, 0.2, 0.3, 0.5, 1], # helps for inbalanced classes as we have, maybe something like a learning rate?
            'max_depth': [3, 4, 5, 10],
            'verbose': [1],
            'random_state' : [42],
            'subsample': [0.25, 0.5, 0.75, 1],
            'min_samples_leaf': [1, 2, 3, 10],
            'max_features': [0.25, .5, .75, 1]
            } 

# safe params into a list of params to loop over
param_grids = [params_gb]

# set up scores
scores = {'gb': None}

# save trained clfs
trained_clfs= []

#%% loop over params grid, takes approx. 2-3 hours
for param_grid, clf, key in zip(param_grids, clfs, scores.keys()):
    gridsearch = GridSearchCV(estimator = clf, 
                              param_grid = param_grid,
                              scoring = 'neg_mean_absolute_error',
                              n_jobs = -1,
                              cv = 5,
                              refit = True,
                              verbose = 1)
      
    # perform gridsearch on train data
    gridsearch.fit(x_train, y_train)
    
    # refit best estimator
    gridsearch.best_estimator_.fit(x_train, y_train)
    
    # safe this estimator
    joblib.dump(gridsearch.best_estimator_, p_loc + f'results/04_models/best_estimator/{key}_{tday}.pkl')
    
    # safe scores in scores dict
    scores[key] = gridsearch.best_score_
    
    # safe trained clf to a list
    trained_clfs.append(gridsearch.best_estimator_)
    
    # safe gridsearch results to dataframe
    pd.DataFrame(gridsearch.cv_results_).to_csv(p_loc + f'results/04_models/best_estimator/gb_regressor/{key}_{tday}.csv')
    
    # safe classification report to excel
    y_pred = np.array(gridsearch.best_estimator_.predict(x_test))

    # cut off y_pred to (0,1)
    y_pred = [0 if pred <= 0 else pred for pred in y_pred]
    y_pred = [1 if pred >= 1 else pred for pred in y_pred]

    report = regression_report(y_test, y_pred)
    report.to_csv(p_loc + f'results/06_reports/regression_reports/{key}_{tday}.csv')