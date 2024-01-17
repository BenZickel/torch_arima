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
from ARIMA.examples.utils import load_data, calc_percentiles, plot

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
    ##########################################
    # Fit model to data and show predictions #
    ##########################################
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

    #########################################################
    # Show effect of amount of training data on predictions #
    #########################################################
    ratios = [1,1/2,1/4,1/8][::-1]
    models = []
    indices = []
    guides = []
    samples = []
    for ratio in ratios:
        n = len(observations)
        indices.append(range(round((1 - ratio)*n), n))
        models.append(create_model([*range(len([*indices[-1]]))], num_predictions))
        guides.append(fit(models[-1], observations[indices[-1]]))
        samples.append(Predictive(models[-1],
                                  guide=guides[-1],
                                  num_samples=num_samples,
                                  return_sites=("_RETURN",))()['_RETURN'])

    plt.figure()
    colors = ['r', 'g', 'b', 'y'][::-1]
    cis = []
    one_year_mean_ci = []
    five_year_mean_ci = []
    for ratio, idx, sample, color in zip(ratios, indices, samples, colors):
        cis.append(calc_percentiles(sample, [0.05, 0.95]))
        plot(year[idx], cis[-1], observations[idx],
             samples_label='Bayesian Estimator at {}% of data at 90% CI'.format(100 * ratio), samples_color=color)
        ci = cis[-1][:,-num_predictions:]
        one_year_mean_ci.append((ci[1]-ci[0])[:12].mean())
        five_year_mean_ci.append((ci[1]-ci[0]).mean())

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Bayesian Predictions at Various Ratios of Observed Data')
    plt.legend(loc='lower left')
    plt.grid()

    output_file_name = plots_dir + '/bayesian_example_ratio.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    plt.figure()
    plt.plot(ratios, one_year_mean_ci, 'bo-', label='One Year Mean 90% CI')
    plt.plot(ratios, five_year_mean_ci, 'ro-', label='Five Year Mean 90% CI')
    plt.xlabel('Ratio of Observed Data')
    plt.ylabel('Mean 90% CI')
    plt.title('Bayesian Estimator 90% CI vs Ratio of Observed Data')
    plt.legend(loc='upper right')
    plt.grid()

    output_file_name = plots_dir + '/bayesian_example_ratio_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
