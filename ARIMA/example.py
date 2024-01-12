'''
Time series forecasting example in PyTorch.

Based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.

Raw data downloaded from https://www.key2stats.com/data-set/view/764.
'''
import pandas as pd
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import os
from ARIMA import ARIMA

# Load data and create model
data = pd.read_csv(os.path.dirname(__file__) + '/data/Monthly_corticosteroid_drug_sales_in_Australia_from_July_1991_to_June_2008.csv')
year = data['time'].to_numpy()
observations = pt.Tensor(data['value'])
model = ARIMA(3,0,1,0,1,2,12)

# Optimize model parameters
optimizer = pt.optim.Adam(model.parameters(), lr=0.05)
for count in range(100):
    optimizer.zero_grad()
    innovations = model(observations, invert=True)
    variance = (innovations**2).mean()
    variance.backward()
    optimizer.step()

# Plot predictions
innovations = model(observations, invert=True)
plt.plot(year, observations, 'b', label='Observations')
simulations = []
num_predictions = 5 * 12
all_year = np.concatenate((year, year[-1] + (np.arange(num_predictions) + 1) * np.diff(year).mean()))
for count in range(100):
    new_innovations = pt.randn(num_predictions) * variance.sqrt()
    all_innovations = pt.cat([innovations, new_innovations])
    all_observations = model(all_innovations).detach().numpy()
    simulations.append(all_observations)
    idx = range(len(innovations) - 1, len(all_innovations))
    plt.plot(all_year[idx], all_observations[idx], 'r', alpha=0.1)
plt.plot(all_year[idx], np.array(simulations).mean(0)[idx], 'r', label='Predictions')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Monthly corticosteroid drug sales in Australia')
plt.legend()
plt.grid()
output_file_name = os.path.dirname(__file__) + '/example.png'
plt.savefig(output_file_name)
print('Saved output file ' + output_file_name)

