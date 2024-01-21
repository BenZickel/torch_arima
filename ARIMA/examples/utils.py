
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pandas as pd
import os

plots_dir = os.path.dirname(__file__) + '/plots'
os.makedirs(plots_dir, exist_ok=True)

def load_data():
    data = pd.read_csv(os.path.dirname(__file__) + '/data/Monthly_corticosteroid_drug_sales_in_Australia_from_July_1991_to_June_2008.csv')
    year = data['time'].to_numpy()
    observations = pt.Tensor(data['value'])
    return year, observations

def calc_percentiles(samples, percentiles):
    samples = np.sort(samples, axis=0)
    indices = [int(np.floor(p * samples.shape[0])) for p in percentiles]
    return samples[indices]
