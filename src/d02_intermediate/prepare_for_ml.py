# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:02:56 2021

@author: joa24jm

This file reads in the df merged_users and converts it into a ML readable dataframe
with no missing values and an equal split for the target (upsampling)
"""
p_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'
#%% imports
import pandas as pd
import utilities as u

#%% read in df
df = pd.read_csv(p_loc + 'data/02_intermediate/merged_users.csv',
                 index_col = 'Unnamed: 0', parse_dates = ['4', '9', 'fup_answer_from', 'bl_answer_from'],na_values = ['??.??.????', '27.02.2522']
                )

#%% define the features
features_list = []

#1 Top 10 countries in number of filled out questionnaires
countries_list = df.country.value_counts().iloc[:10].index.tolist()
country_dummy_df = pd.get_dummies(df[df.country.isin(countries_list)]['country'])
df = pd.concat([df, country_dummy_df], axis = 1)
features_list.extend(country_dummy_df.columns.tolist())

#2 The four seasons

# first derive whether the country is northern or not
df['is_northern'] = df.country.apply(u.is_country_northern) # takes a coule of minutes to run

# now derive the season from the answer date and the information, whether the country is northern or not
df['fup_season'] = df.apply(lambda x: u.get_season(x.fup_answer_from, x.is_northern), axis=1)
season_dummy_df = pd.get_dummies(df['fup_season'])
df = pd.concat([df, season_dummy_df], axis = 1)
features_list.extend(season_dummy_df.columns.tolist())

#3 The gender
gender_dummy_df = pd.get_dummies(df['5'], drop_first=True)
df = pd.concat([df, gender_dummy_df], axis = 1)
features_list.extend(gender_dummy_df.columns.tolist())

#4 The year of birth
s = df['4'].dt.year.rename('year_of_birth')
df = pd.concat([df, s], axis = 1)


#4.1 The age derived by birthdate and fup_answer_from
## convert fup_answer and year_of_birth to a timedelta
df['timedelta'] = pd.to_datetime(df['fup_answer_from'].dropna().dt.date) - pd.to_datetime(df['4'].dropna().dt.date)
df['age'] = df['timedelta'].apply(lambda x: round(x.days/365, 1))
features_list.append('age')


#5 questions 4,5,6, and 7 from the daily questionnaire
features_list.extend(['question4', 'question5', 'question6', 'question7'])

#%% shrink the dataset to features and target
df = df.loc[country_dummy_df.index, features_list + ['question1', 'user_id']]

# drop missing values for now -> this leads to a lost of approx 650 users
#TODO: Better approach!
df.dropna(inplace=True)

#%% draw randomly from the stratum with question1 == 0 and add these to the df
# so that question1 becomes equally distributed
times_to_draw = df.question1.value_counts()[1] - df.question1.value_counts()[0]
sampled_df = df[df.question1 == 0].sample(n=times_to_draw, replace = True)
df = df.append(sampled_df, ignore_index = True)

#%% safe dataframe
df.to_csv(p_loc + 'data/03_processed/df_equal_splits_with_age.csv', index = False)
