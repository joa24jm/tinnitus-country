# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:56:24 2021

@author: joa24jm

This file reads in the csv files from data/01_raw
and converts them into pandas dataframes.
The files have been exported from phpmyAdmin using the 'csv for ms excel' option.

Update:
join positions of dataframes matters, beware of left joins and the drop duplicates kwd
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

#%% merge daily with basline on a left join so that the shape[0] of daily stays the same

print(daily.shape)

# merge baseline questionnaire with daily
merged_df = pd.merge(left = daily, right = baseline, left_on='user_id', right_index=True, 
                     how = 'left')

print(merged_df.shape)

#%% drop columns

# drop columns that are not of interest
cols_to_drop = ['notification_date', 'autosaved', 'notification_fixed',
                'save_date', 'updated_at', 'user_id_reference', 'user_agent',
                'soundlevel']

# drop specified columns
merged_df = merged_df.drop(columns=cols_to_drop)

#%% add country information to merged_df and baseline df
print(merged_df.shape)
merged_df = pd.merge(left = merged_df, right = users_meta[['user_id', 'country']], 
                     on = 'user_id', how = 'left')
print(merged_df.shape)

print(baseline.shape)
baseline = pd.merge(baseline, users_meta[['user_id', 'country']],
                    right_on = 'user_id', left_index = True, how = 'left')
print(baseline.shape)


#%% add continent information by reading in continent dataframe
url = 'https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv'
continent_codes = pd.read_csv(url)
# drop na vals for country codes to prevent duplicate matches on merging
continent_codes.dropna(axis = 'index', subset = ['Two_Letter_Country_Code'],
                       inplace = True)
continent_codes.drop_duplicates(subset=['Two_Letter_Country_Code'], 
                                keep='first', inplace=True)
continent_codes.drop
print(merged_df.shape)
# add column by left outer join with merged_df
merged_df = pd.merge(left = merged_df, 
                     right = continent_codes[['Continent_Name', 'Two_Letter_Country_Code', 'Country_Name']], 
                     left_on = 'country', right_on = 'Two_Letter_Country_Code', 
                     how = 'left').drop(columns='Two_Letter_Country_Code')
print(merged_df.shape)

#%% rename columns, fup = FollowUp, bl= Baseline
rename_cols = {'created_at_x':'fup_answer_from',
               'created_at_y': 'bl_answer_from'}
merged_df.rename(columns = rename_cols, inplace= True)


#%% check for matches in tyt plattform from survey and users
# rename col for better readability
long_email = '11. If you said yes to the last question, would you mind to provide us with your e-mail address to give us the opportunity to contact you?'
survey.rename(columns={long_email:'email'}, inplace = True)

# strip() and lower() all mails
survey.email.str.strip().str.lower()



# get user_ids by merge on email
survey = pd.merge(left = survey, right = users[['email', 'id']], how = 'left', on = 'email')

# rename user id column for convenience
survey.rename(columns={'id':'user_id'}, inplace = True)

# replace missing values from nan to -1 to prevent unintentional merging operations (nan from df1 matches nan from df2!)
survey['user_id'] = survey['user_id'].fillna(-1)



# get cols of interest from survey df ('die Informatinen in den Spalten I, J und K')
cols_of_interest = [ '4. Overall speaking, does the use of the TrackYourTinnitus App helped you?',
       '5. Would you recommend the TrackYourTinnitus App to other users?',
       '6. Did you perceive any adverse events when using the TrackYourTinnitus App?']

# get information of these users from merged
merged_df = pd.merge(left = merged_df, right = survey[cols_of_interest + ['user_id']], on = 'user_id', how = 'left')


#%% save data from survey users and merged_df with all users

# save merged_df
merged_df.to_csv(proj_loc + 'data/02_intermediate/merged_users_with_survey_data.csv',
                    index = True)

