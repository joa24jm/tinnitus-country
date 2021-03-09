# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:12:03 2021

@author: joa24jm

This file reads in the best model (i.e. gradient boosting machine) and calculates
a permutation importance on the dataset to find out about the best feature.

"""

p_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'

#%% imports
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.inspection import permutation_importance

#%% read in df
df = pd.read_csv(p_loc + 'data/03_processed/df_equal_splits.csv')

# read in model
clf = joblib.load('results/04_models/best_estimator/gb_21_03_08_14_12.pkl')

#%% train test split
features = ['AT', 'CA', 'CH','DE','GB', 'IT', 'NL', 'NO', 'RU', 'US', # countries
            'autumn', 'spring', 'summer', 'winter',                  # season
            # 'Male', 'year_of_birth',                                 # demographics
            'question4', 'question5', 'question6', 'question7'
            ]      # EMAs

X = df[features] # all columns except for the last
y = df['question1']  # last col as target


# split up data into train and test, stratify on y, set random_state and shuffle
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42,
                                                    shuffle = True,
                                                    stratify = y)

#%% calculate permutation importance on test data
r = permutation_importance(clf, x_test, y_test,
                           n_repeats=30,
                           random_state=0)


for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X.columns[i]:<8}\t\t\t"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")


#%% save results
