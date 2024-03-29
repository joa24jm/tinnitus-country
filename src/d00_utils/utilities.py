# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:41:01 2021

@author: joa24jm

Helper function for this project

"""

import pandas as pd

def unroll_baseline(ans):
    """
    Convert answer dataframe with one column for all answers of all questions
    into a dataframe that has one column per question containing the answers
    Additionally, safe the date when this user filled out the questionnaire

    Parameters
    ----------
    ans : dataframe with answers from baseline questionnaire

    Returns
    -------
    unrolled answers with user_id as index and one column per question

    """
    import pandas as pd

    list_of_question_ids = ans['question_id'].dropna().unique()
    list_of_users = [int(u) for u in ans['user_id'].dropna().unique()]
    
    
    # result dataframe with user_id as index
    answers_per_user = pd.DataFrame(data = None, 
                                    index = list_of_users)
    
    # turn one column for all questions into one column per question
    for q_id in list_of_question_ids:
        # get all answers of all users for this question id
        res = ans[ans.question_id == q_id][['user_id', 'answer']]
        # drop users that has na values for their user_id
        res = res[res['user_id'].notna()]
        # convert user_ids to int32
        res = res.astype({'user_id':'int32'})
        # drop duplicates and keep the first user_id
        res.drop_duplicates(subset=['user_id'], inplace = True)
        # use the user_id as an index
        res.set_index('user_id', drop = True, inplace = True)
        # rename column to q_id
        res.rename(columns = {'answer': f'{int(q_id)}'},inplace = True)
        # concat res to answer_per_user on user_id
        answers_per_user = pd.concat([answers_per_user, res], axis = 1)

    # safe the date for each user
    date = ans.drop_duplicates(subset = ['user_id'])[['user_id', 'created_at']]
    # drop users that has na values for their user_id
    date = date[date['user_id'].notna()]
    # use the user_id as an index
    date.set_index('user_id', drop = True, inplace = True)
    # concat with answers_per_user
    answers_per_user = pd.concat([answers_per_user, date], axis = 1)
    
    return answers_per_user

def remove_testusers(df, column = 'user_id'):
    """
    Removes all rows in dataframe that contains a testuser

    Parameters
    ----------
    df : dataframe with user_id as index or column
    column : Name of user_id column. The default is 'user_id'.

    Returns
    -------
    Cleaned dataframe without test_users

    """

    testusers = [553, 2186, 2244, 2242, 1563, 1119, 845, 374, 115, 54, 
                 1, 2, 11, 49, 36, 48, 461, 41, 39, 553, 728, 43, 47, 64, 66]
    
    # if user_id is a column
    try: 
        df = df[~df[f'{column}'].isin(testusers)]
    
    except:
        
        # if user_id is an index
        try: 
            bad_df = df.index.isin(testusers)
            df = df[~bad_df]
        
        # this should never raise   
        except: 
            raise ValueError('user_id is no column and no index.')
    
    return df

def show_values_on_bars(axs, h_v="v", space=0.4, normalize = False):
    import numpy as np
    import matplotlib
    """
    Parameters
    ----------
    axs : matplotlib axes object.
    h_v : Whether the barplot is horizontal or vertical. 
          "h" represents the horizontal barplot, 
          "v" represents the vertical barplot. .
    space : The space between value text and the top edge of the bar. 
            Only works for horizontal mode. The default is 0.4.
    normalize: Whether to plot on normalized values or not

    Returns
    -------
    None.

    """
    matplotlib.rcParams.update({'font.size': 8})
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = (p.get_y() + p.get_height())*1.02
                if normalize:    
                    value = round(p.get_height(), 3)
                else:
                    value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                if normalize:    
                    _x = p.get_x() + p.get_width()
                    value = p.get_width()
                    float_formatter = '{:.2%}'.format
                    ax.text(_x, _y, float_formatter(value), ha="left")
                else:
                    value = int(p.get_width())
                    ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        

# define a functions, that returns true if the country is from the northern_hemisphere
def is_country_northern(country):
    """
    country: ISO2 string, i.e. 'DE', 'US'
    Returns True if the country is from the northern hemisphere
    """
    import pandas as pd
    p_loc = 'C:/Users/joa24jm/Documents/tinnitus-country/'
    southern_countries = pd.read_csv(p_loc + 'data/00_sources/southern_countries.csv').iso2.values
    
    if country not in southern_countries:
        if isinstance(country, str):
            return True
        else:
            import numpy as np
            return np.nan
        
    if country in southern_countries:
        return False

def get_season(d, is_northern):
    """
    Returns the season given a datetime object

    Parameters
    ----------
    date : datetime object
    is_northern: boolean indicating whether the country is from the northern hemisphere
    Returns
    -------
    season as string.

    """
    from datetime import date, datetime

    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    if is_northern:
        seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
                    ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
                    ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
                    ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
                    ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
    else:
        seasons = [('summer', (date(Y,  1,  1),  date(Y,  3, 20))),
            ('autumn', (date(Y,  3, 21),  date(Y,  6, 20))),
            ('winter', (date(Y,  6, 21),  date(Y,  9, 22))),
            ('spring', (date(Y,  9, 23),  date(Y, 12, 20))),
            ('summer', (date(Y, 12, 21),  date(Y, 12, 31)))]
    

    if isinstance(d, datetime):
        d = d.date()
    d = d.replace(year=Y)
    try:
        return_value = next(season for season, (start, end) in seasons
                    if start <= d <= end)
    except:
        return_value = None
    return return_value

def format_ct(ct):
    """
    

    Parameters
    ----------
    ct : Unnormalized dataframe with values

    Returns
    -------
    ct_return : columns-Normalized dataframe with format {value} (%)
    """

    ct_return = ct.copy()
    for col in ct.columns:
        s = ct[col].values.sum()
        for idx in ct[col].index:
            val = ct.loc[idx, col]
            ct_return.loc[idx, col] = (f'{val} ({val/s:.1%})')

    return ct_return

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    import scipy.stats as ss
    import numpy as np
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def regression_report(y_true, y_pred):
    """
    Print a regression report for your regression problem, as provided from:
    https://github.com/scikit-learn/scikit-learn/issues/18454

    Parameters
    ----------
    y_true - Ground truth of our regression
    y_pred - Estimated values for the regression problem

    Returns
    -------
    df containing the metrics
    """

    # import libraries
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, \
        explained_variance_score
    import numpy as np
    import pandas as pd

    metrics = {
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'explained_variance_score': explained_variance_score(y_true, y_pred)
    }

    df = pd.DataFrame(metrics, index = [0])

    return df