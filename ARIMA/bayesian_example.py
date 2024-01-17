'''
Time series Bayesian estimator forecasting example based on PyTorch and Pyro.

Based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.

Raw data downloaded from https://www.key2stats.com/data-set/view/764.
'''
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pyro
import os
from pyro.infer import Predictive
from ARIMA import BayesianARIMA
from ARIMA.example import load_data, calc_percentiles

def create_model(obs_idx, num_predictions):
    # Create model with non-overlapping observed and predicted sample indices.
    predict_idx = [*range(max(obs_idx) + 1 + num_predictions)]
    predict_idx = [idx for idx in predict_idx if idx not in obs_idx]
    return BayesianARIMA(3, 0, 1, 0, 1, 2, 12, obs_idx=obs_idx, predict_idx=predict_idx)

def fit(model, observations,
        lr_sequence=[(0.005, 100),
                     (0.010, 100)] * 5 +
                    [(0.005, 100),
                     (0.001, 100)]):
    # Create posterior for Bayesian model
    guide = pyro.infer.autoguide.guides.AutoMultivariateNormal(model)
    guide(observations)
    guide.loc.data[:] = 0
    for lr, num_iter in lr_sequence:
        optimizer = pyro.optim.Adam(dict(lr=lr))
        svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO(num_particles=5))
        for count in range(num_iter):
            svi.step(observations)
    return guide

if __name__ == "__main__":
    year, observations = load_data()

    num_predictions = 5 * 12
    obs_idx = range(len(observations))

    # num_predictions = 0
    # obs_idx = [*range(80)] + [*range(140,len(observations))]

    # num_predictions = 0
    # obs_idx = range(60, len(observations))

    model = create_model(obs_idx, num_predictions)

    guide = fit(model, observations[model.obs_idx])

    # Make predictions
    num_samples = 1000
    predictive = Predictive(model,
                            guide=guide,
                            num_samples=num_samples,
                            return_sites=("_RETURN",))
    samples = predictive()['_RETURN']

    # Plot predictions
    plt.plot(year[model.obs_idx], observations[model.obs_idx], 'b', label='Observations')
    all_year = np.concatenate((year, year[-1] + (np.arange(len(model.obs_idx) + len(model.predict_idx) - len(year)) + 1) * np.diff(year).mean()))
    idx = sorted(set(np.clip([min(model.predict_idx) - 1, max(model.predict_idx) + 1] + model.predict_idx, 0, len(all_year) - 1)))
    ci = calc_percentiles(samples[:,idx], [0.05, 0.95])
    plt.fill_between(all_year[idx], ci[0], ci[1], color='r', alpha=0.5, label='Bayesian Model Estimates at 90% CI')
    
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Monthly corticosteroid drug sales in Australia')
    plt.legend(loc='upper left')
    plt.grid()

    plots_dir = os.path.dirname(__file__) + '/plots'
    os.makedirs(plots_dir, exist_ok=True)

    output_file_name = plots_dir + '/bayesian_example.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
