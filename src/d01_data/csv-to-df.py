# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:56:24 2021

@author: joa24jm

This file reads in the JSON files from data/02_intermediate
and converts them into a pandas dataframe.
The json files in 01_raw contain a list [] with meta data in the beginning.
These lists have been removed and the files have been saved in 02_intermediate.
"""

proj_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'

#%% imports
import pandas as pd

#%% read in dataframes
ans =   pd.read_csv(proj_loc + 'data/01_raw/answers.csv', delimiter = ';')
qns =   pd.read_csv(proj_loc + 'data/01_raw/questions.csv', delimiter = ';')
sdans = pd.read_csv(proj_loc + 'data/01_raw/standardanswers.csv', delimiter = ';')
users = pd.read_csv(proj_loc + 'data/01_raw/users.csv', delimiter = ';')
users_meta = pd.read_csv(proj_loc + 'data/01_raw/users_metadata.csv', delimiter = ';')

#%% parse answers df and save every question into a new column
list_of_question_ids = ans['question_id'].dropna().unique()
list_of_users = ans['user_id'].dropna().unique()

answers_per_user = pd.DataFrame(data = None, 
                                index = list_of_users, 
                                columns = list_of_question_ids)

for q_id in list_of_question_ids:
    res = ans[ans.question_id == q_id][['user_id', 'answer']]
    s = pd.Series()