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
daily = pd.read_csv(proj_loc + 'data/01_raw/standardanswers.csv', delimiter = ';',
                    parse_dates=['created_at', 'save_date'])
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
                'save_date', 'updated_at', 'user_id_reference', 'user_agent',
                'soundlevel']

# drop specified columns
merged_df = merged_df.drop(columns=cols_to_drop)

# set answer_id as index
merged_df.set_index('id', drop = True, inplace = True)

# add country information
merged_df = pd.merge(merged_df, users_meta[['user_id', 'country']], 
                     on = 'user_id', how = 'left')
baseline = pd.merge(baseline, users_meta[['user_id', 'country']],
                    right_on = 'user_id', left_index = True, how = 'left')

# add continent information by reading in continent dataframe
url = 'https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv'
continent_codes = pd.read_csv(url)

# add column by left outer join with merged_df
merged_df = pd.merge(merged_df, continent_codes[['Continent_Name', 'Two_Letter_Country_Code', 'Country_Name']], 
                     left_on = 'country', right_on = 'Two_Letter_Country_Code', 
                     how = 'left').drop(columns='Two_Letter_Country_Code')

# rename columns, fup = FollowUp, bl= Baseline
rename_cols = {'created_at_x':'fup_answer_from',
               'created_at_y': 'bl_answer_from'}
merged_df.rename(columns = rename_cols, inplace= True)

# check for matches in tyt plattform from survey and users
survey_emails = [mail.lower() for mail in survey.iloc[:, -2].dropna().unique()
                 if '@' in mail]

# rename col for better readability
survey.rename(columns={'11. If you said yes to the last question, would you mind to provide us with your e-mail address to give us the opportunity to contact you?':'mail'}, inplace = True)

# get cols of interest from survey df ('die Informatinen in den Spalten I, J und K')
cols_of_interest = [ '4. Overall speaking, does the use of the TrackYourTinnitus App helped you?',
       '5. Would you recommend the TrackYourTinnitus App to other users?',
       '6. Did you perceive any adverse events when using the TrackYourTinnitus App?',
       'mail']

# get mails and cols of interest of that survey df
survey_cache = survey[survey.loc[:,'mail'].isin(survey_emails)][cols_of_interest]

# merge with df 'users' to get user_ids from emails
survey_merged = pd.merge(survey_cache, users, left_on='mail', right_on = 'email')[cols_of_interest + ['id']]

# get information of these users from merged
cache = pd.merge(merged_df, survey_merged, how='outer', left_on='user_id', right_on='id')
# cache = cache[cache.mail.notna()]

# save mapping from id to mail
id_mail = cache[['id', 'mail']].drop_duplicates()

# drop mail and id from cache_df
cache.drop(axis = 1, labels = ['id', 'mail'], inplace = True)


#%% save data from survey users and merged_df with all users

# save data from survey users
cache.to_csv(proj_loc + 'data/02_intermediate/survey_users.csv',
                    index = False)

# save id_mail mapping seperately for data protection
id_mail.to_csv(proj_loc+ 'data/02_intermediate/survey_id_mail.csv',
                    index = False)

# save merged_df
merged_df.to_csv(proj_loc + 'data/02_intermediate/merged_users.csv',
                    index = True)

# safe baseline df
baseline.to_csv(proj_loc+ 'data/02_intermediate/baseline.csv',
                    index = True)
