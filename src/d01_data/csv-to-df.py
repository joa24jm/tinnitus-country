# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:56:24 2021

@author: joa24jm

This file reads in the csv files from data/01_raw
and converts them into a pandas dataframe.
The files have been exported from phpmyAdmin using the 'csv for ms excel' option.
"""

proj_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'

#%% imports
import pandas as pd
import utilities as utils

#%% read in dataframes
baseline =   pd.read_csv(proj_loc + 'data/01_raw/answers.csv', delimiter = ';')
qns =   pd.read_csv(proj_loc + 'data/01_raw/questions.csv', delimiter = ';')
daily = pd.read_csv(proj_loc + 'data/01_raw/standardanswers.csv', delimiter = ';')
users = pd.read_csv(proj_loc + 'data/01_raw/users.csv', delimiter = ';')
users_meta = pd.read_csv(proj_loc + 'data/01_raw/users_metadata.csv', delimiter = ';')
survey = pd.read_csv(proj_loc + 'data/01_raw/umfrageonline-1277718.csv', delimiter = ';')

#%% parse answers df and save every question into a new column
baseline = utils.unroll_baseline(baseline)

# remove testusers
baseline = utils.remove_testusers(baseline)
daily = utils.remove_testusers(daily)
users = utils.remove_testusers(users, column = 'id')
users_meta = utils.remove_testusers(users_meta)

# merge baseline questionnaire with daily
merged_df = pd.merge(daily, baseline, left_on='user_id', right_index=True, how = 'right')

# drop columns that are not of interest
cols_to_drop = ['notification_date', 'autosaved', 'notification_fixed',
                'created_at_x', 'updated_at', 'user_id_reference', 'user_agent',
                'soundlevel']

# drop specified columns
merged_df = merged_df.drop(columns=cols_to_drop)

# set answer_id as index
merged_df.set_index('id', drop = True, inplace = True)

# add country information
merged_df = pd.merge(merged_df, users_meta[['user_id', 'country']], 
                     on = 'user_id', how = 'left')

# add continent information by reading in continent dataframe
url = 'https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv'
continent_codes = pd.read_csv(url)

# add column by left outer join with merged_df
merged_df = pd.merge(merged_df, continent_codes[['Continent_Name', 'Two_Letter_Country_Code', 'Country_Name']], 
                     left_on = 'country', right_on = 'Two_Letter_Country_Code', 
                     how = 'left').drop(columns='Two_Letter_Country_Code')

# rename columns, fup = FollowUp, bl= Baseline
rename_cols = {'save_date':'fup_answer_from',
               'created_at_y': 'bl_answer_from'}
merged_df.rename(columns = rename_cols, inplace= True)

# check for matches in tyt plattform from survey and users
# Spalte 'O' muss ein ja drinstehen
survey_filtered = survey[survey.iloc[:, -3] == 'ja']
survey_emails = [mail.lower() for mail in survey_filtered.iloc[:, -2].dropna().unique()
                 if '@' in mail]

# get user_ids of these mail addresses
user_ids = users[users['email'].isin(survey_emails)].id.values

# get information of these users from merged
survey_users = merged_df[merged_df['user_id'].isin(user_ids)]
# convert user_id to integer
survey_users = survey_users.astype({'user_id':'int32'})

#%% save data from survey users and merged_df with all users
survey_users.to_csv('data/02_intermediate/survey_users.csv',
                    index = False)

merged_df.to_csv('data/02_intermediate/merged_users.csv',
                    index = True)

baseline.to_csv('data/02_intermediate/baseline.csv',
                    index = True)
