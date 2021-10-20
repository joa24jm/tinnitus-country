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
import joblib
from sklearn.metrics import confusion_matrix
import datetime

#%% read in df

df = pd.read_csv(p_loc + 'data/03_processed/df_equal_splits_with_age.csv')

#%% select features and target
features = ['question4', 'question5', 'question6', 'question7']      # EMAs

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

# safe date of approach
tday = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')

clfs = [GradientBoostingClassifier()]

params_gb = {'learning_rate': [0.5], # helps for inbalanced classes as we have, maybe something like a learning rate?
            'max_depth': [10],
            'verbose': [1],
            'random_state' : [42],
            'subsample': [1],
            'min_samples_leaf': [1],
            'max_features': [.75]
            }

#GradientBoostingRegressor(learning_rate=0.5, max_depth=10, max_features=0.75,
                         #  random_state=42, subsample=1, verbose=1)

# safe params into a list of params to loop over
param_grids = [params_gb]

# set up scores
scores = {'gb': None}

# save trained clfs
trained_clfs= []

#%% loop over params grid
for param_grid, clf, key in zip(param_grids, clfs, scores.keys()):
    gridsearch = GridSearchCV(estimator = clf, 
                              param_grid = param_grid,
                              scoring = 'accuracy',
                              n_jobs = -1,
                              cv = 5,
                              refit = True,
                              verbose = 2)
      
    # perform gridsearch on train data
    gridsearch.fit(x_train, y_train)
    
    # refit best estimator
    gridsearch.best_estimator_.fit(x_train, y_train)

    # print feature importance
    print(gridsearch.best_estimator_.feature_importances_)
    
    # safe this estimator
    joblib.dump(gridsearch.best_estimator_, p_loc + f'results/04_models/best_estimator/{key}_{tday}.pkl')
    
    # safe scores in scores dict
    scores[key] = gridsearch.best_score_
    
    # safe trained clf to a list
    trained_clfs.append(gridsearch.best_estimator_)
    
    # safe gridsearch results to dataframe
    pd.DataFrame(gridsearch.cv_results_).to_csv(p_loc + f'results/04_models/gridsearch/{key}_{tday}.csv')
    
    # safe classification report to excel
    y_pred = gridsearch.best_estimator_.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, target_names = ['Tinnitus NO', 'Tinnitus YES'])
    pd.DataFrame(report).transpose().to_csv(p_loc + f'results/06_reports/classification_reports/{key}_{tday}.csv')
    
    # safe confusion matrix
    labels = ['Tinnitus NO', 'Tinnitus YES']
    pd.DataFrame(confusion_matrix(y_test, y_pred), index = labels, 
                 columns = labels).to_csv(p_loc + f'results/06_reports/confusion_matrices/confusion_{key}_{tday}.csv')
    
#%% plot confusion matrix
import seaborn as sns

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True,
            fmt = 'd', cmap = 'Blues')


