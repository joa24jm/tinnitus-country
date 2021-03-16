# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:07:29 2021

@author: joa24jm

Pre-Process temp by month and country df

"""
#%% set up
# project location
p_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'

import pandas as pd
import numpy as np
import re

#%% read in the temperature df
t = pd.read_csv(p_loc + '/data/01_raw/temp_by_country_by_month.csv')

#%% shape data to floats
target_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
               'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Year']
for col in target_cols:
    for i, val in enumerate(t[col].str.split('(')):
        t.loc[i, col] = val[0]

#%% convert object to numerics

# convert unicode en-dash to an ASCII hyphen for detection of negative values
# using regular expressions
for col in target_cols:
    t[col] = t[col].apply(lambda s: re.sub(r'[^\x00-\x7F]+','-', s)).astype('float')

#%% if a country is represented by more than one city, create the mean of the values columnwise
t = t.groupby('Country').mean()

#%% get the country code for each country

# read in country codes as cc
url = 'https://pkgstore.datahub.io/core/country-list/data_csv/data/d7c9d7cfb42cb69f4422dec222dbbaa8/data_csv.csv'
cc = pd.read_csv(url)

# merge with 't' df
t = pd.merge(t, cc, left_on = 'Country', right_on = 'Name')

#%% export 
t.to_csv(p_loc + '/data/02_intermediate/temp_by_country-code_by_month.csv')

















