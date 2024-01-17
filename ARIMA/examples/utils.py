
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pandas as pd
import os

def load_data():
    data = pd.read_csv(os.path.dirname(__file__) + '/data/Monthly_corticosteroid_drug_sales_in_Australia_from_July_1991_to_June_2008.csv')
    year = data['time'].to_numpy()
    observations = pt.Tensor(data['value'])
    return year, observations

def calc_percentiles(samples, percentiles):
    samples = np.sort(samples, axis=0)
    indices = [int(np.floor(p * samples.shape[0])) for p in percentiles]
    return samples[indices]

def plot(time, samples, observations,
         samples_label=None, observations_label=None,
         samples_color='r', observations_color='b'):
    if observations_label is not None:
        plt.plot(time, observations, color=observations_color, label='Observations')
    all_time = np.concatenate((time, time[-1] + (np.arange(samples.shape[1] - len(time)) + 1) * np.diff(time).mean()))
    idx = range(len(observations) - 1, len(all_time))
    if samples_label is not None:
        plt.fill_between(all_time[idx], samples[0, idx], samples[1, idx], color=samples_color, alpha=0.5,
                         label=samples_label)

