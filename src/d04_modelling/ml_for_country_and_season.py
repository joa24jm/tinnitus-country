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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#%% read in df
df = pd.read_csv(p_loc + 'data/03_processed/df_equal_splits.csv')

#%% select features and target
features = ['AT', 'CA', 'CH','DE','GB', 'IT', 'NL', 'NO', 'RU', 'US', # countries
            'autumn', 'spring', 'summer', 'winter',                  # season
            'Male', 'year_of_birth',                                 # demographics
            'question4', 'question5', 'question6', 'question7']      # EMAs

X = df[features] # all columns except for the last
y = df['question1']  # last col as target


# split up data into train and test, stratify on y, set random_state and shuffle
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42,
                                                    shuffle = True,
                                                    stratify = y)

#%% prepare gridsearch for both random forest (rf) and gradient boosting classifier (gb)
# list of default classifiers
clfs = [RandomForestClassifier(), GradientBoostingClassifier()]

# set up params for grid search
params_rf = {'bootstrap': [True, False], # False means that we use all samples
            'criterion': ['gini', 'entropy'], # splitting criterion as discussed in lecture
            'max_depth': [3, 4, 5, 10], # depth of a single tree
            'max_features': [0.25, .5, .75, 1], # fraction of features used when building a new forest
            'min_samples_leaf': [1, 2, 3, 10], # min number of samples at a final leaf
            'n_estimators': [10, 50, 100, 200], # No. of trees in the forest
            'n_jobs': [-1], # use all processors on cpu
            'random_state': [42],
            'verbose': [1], # more information provided during the process
            }

params_gb = {'learning_rate': [0.1, 0.2, 0.3, 0.5, 1], # helps for inbalanced classes as we have, maybe something like a learning rate?
            'max_depth': [3, 4, 5, 10],
            'verbose': [1],
            'random_state' : [42],
            'subsample': [0.25, 0.5, 0.75, 1],
            'min_samples_leaf': [1, 2, 3, 10],
            'max_features': [0.25, .5, .75, 1]
            } 

# safe params into a list of params to loop over
param_grids = [params_rf, params_gb]

# set up scores
scores = {'rf': None, 'gb': None}

# save trained clfs
trained_clfs= []

#%% loop over params grid
for param_grid, clf, key in zip(param_grids, clfs, scores.keys()):
    gridsearch = GridSearchCV(estimator = clf, 
                              param_grid = param_grid,
                              scoring = 'f1',
                              n_jobs = -1,
                              cv = 5,
                              refit = True,
                              verbose = 3)
      
    # perform gridsearch on train data
    gridsearch.fit(x_train, y_train)
    
    # safe scores in scores dict
    scores[key] = gridsearch.best_score_
    
    # safe trained clf to a list
    trained_clfs.append(gridsearch.best_estimator_)

#%% get classification reports for best estimators
# save both clf reports into a list
clf_reports = []

# loop over trained_clfs
for trained_clf in trained_clfs:
    clf_reports.append(classification_report(trained_clf.predict(x_test), y_test))
