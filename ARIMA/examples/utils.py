
import datetime
import torch as pt
import numpy as np
import pandas as pd
import os

plots_dir = os.path.dirname(__file__) + '/plots'
os.makedirs(plots_dir, exist_ok=True)

def load_mortality_data():
    # Load data and set time as index
    data = pd.read_csv(os.path.dirname(__file__) + '/data/GBRTENWstmfout.csv')
    data = data.set_index(['Year', 'Week'])
    male_deaths = data[data['Sex']=='m']['Total']
    female_deaths = data[data['Sex']=='f']['Total']
    # Calculate cummulative deaths
    data = pd.DataFrame(data={'Male Deaths': male_deaths, 'Female Deaths': female_deaths}).sort_index()
    for col in ['Female Deaths', 'Male Deaths']:
        data[col] = data[col].cumsum()
    data['Date'] = pd.to_datetime([datetime.date.fromisocalendar(*index, 7) for index in data.index])
    data = data.set_index(['Date'])
    # Interpolate to monthly cummulative deaths
    time = data.index.to_julian_date()
    month = data.index.to_period('M')
    new_time = month[0] + np.arange(1, (month[-1] - month[0]).n)
    new_time = [value.to_timestamp().to_julian_date() for value in new_time]
    data['Date'] = time
    data = data.set_index('Date')
    data = data.reindex(pd.unique(sorted(list(data.index)) + new_time))
    data = data.interpolate('index')
    data = data.reindex(new_time)
    data.index = pd.to_datetime(data.index, origin='julian', unit='D')
    # Return monthly deaths
    return data.diff()[1:]

def load_data():
    data = pd.read_csv(os.path.dirname(__file__) + '/data/Monthly_corticosteroid_drug_sales_in_Australia_from_July_1991_to_June_2008.csv')
    year = data['time'].to_numpy()
    observations = pt.Tensor(data['value'])
    return year, observations

def calc_percentiles(samples, percentiles):
    samples = np.sort(samples, axis=0)
    indices = [int(np.floor(p * samples.shape[0])) for p in percentiles]
    return samples[indices]

def moving_sum(data, window_len, dim):
    select = [slice(None)] * len(data.shape)
    ret_val = 0
    for start_idx in range(window_len):
        select[dim] = slice(start_idx, data.shape[dim] - window_len + start_idx + 1)
        ret_val += data[select]
    return ret_val
