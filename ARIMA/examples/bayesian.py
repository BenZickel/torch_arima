'''
Time series Bayesian estimator forecasting example based on PyTorch and Pyro.

Based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.

Raw data downloaded from https://www.key2stats.com/data-set/view/764.
'''
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pyro
from pyro.infer import Predictive
from ARIMA import BayesianARIMA
from ARIMA.examples.utils import load_data, calc_percentiles, plots_dir, timeit
from ARIMA.pyro_utils import Importance
from ARIMA.examples import __name__ as __examples__name__

def create_model(obs_idx, num_predictions):
    # Create model with non-overlapping observed and predicted sample indices.
    predict_idx = [*range(max(obs_idx) + 1 + num_predictions)]
    predict_idx = [idx for idx in predict_idx if idx not in obs_idx]
    return BayesianARIMA(3, 0, 1, 0, 1, 2, 12, obs_idx=obs_idx, predict_idx=predict_idx)

@timeit
def fit(model, observations,
        lr_sequence=[(0.005, 100),
                     (0.010, 100)] * 5 +
                    [(0.005, 100),
                     (0.001, 100)],
        loss=pyro.infer.JitTrace_ELBO,
        loss_params=dict(num_particles=20, vectorize_particles=True, ignore_jit_warnings=True)):
    # Create posterior for Bayesian model
    guide = pyro.infer.autoguide.guides.AutoMultivariateNormal(model)
    guide(observations)
    guide.loc.data[:] = 0
    loss = loss(**loss_params)
    for lr, num_iter in lr_sequence:
        optimizer = pyro.optim.Adam(dict(lr=lr))
        svi = pyro.infer.SVI(model, guide, optimizer, loss=loss)
        for count in range(num_iter):
            svi.step(observations)
    return guide

if __name__ == '__main__' or __examples__name__ == '__main__':
    ##########################################
    # Fit model to data and show predictions #
    ##########################################
    year, observations = load_data()

    num_predictions = 5 * 12
    obs_idx = range(len(observations))

    model = create_model(obs_idx, num_predictions)

    guide = fit(model, observations[model.obs_idx])

    # Make predictions
    num_samples = 1000
    predictive = Predictive(model.predict,
                            guide=guide,
                            num_samples=num_samples,
                            return_sites=('_RETURN',))
    samples = predictive(observations[model.obs_idx])['_RETURN']

    confidence_interval = [0.05, 0.95]

    plt.figure()

    # Plot observations
    plt.plot(year[model.obs_idx], observations[model.obs_idx], 'b', label='Observations')

    # Plot confidence interval of predictions
    all_year = np.concatenate((year, year[-1] + (np.arange(len(model.obs_idx) + len(model.predict_idx) - len(year)) + 1) * np.diff(year).mean()))
    idx = sorted(set(np.clip([min(model.predict_idx) - 1, max(model.predict_idx) + 1] + model.predict_idx, 0, len(all_year) - 1)))
    ci = calc_percentiles(samples[:,idx], confidence_interval)
    plt.fill_between(all_year[idx], ci[0], ci[1], color='r', alpha=0.5, label='Bayesian Model Estimates at 90% CI')
    
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Monthly corticosteroid drug sales in Australia')
    plt.legend(loc='upper left')
    plt.grid()

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
        samples.append(Predictive(models[-1].predict,
                                  guide=guides[-1],
                                  num_samples=num_samples,
                                  return_sites=("_RETURN",))(observations[indices[-1]])['_RETURN'])

    plt.figure()
    spans = np.array(ratios) * (max(year) - min(year))
    colors = ['r', 'g', 'b', 'y'][::-1]
    cis = []
    one_year_mean_ci = []
    five_year_mean_ci = []
    median = []
    for span, idx, model, sample, color in zip(spans, indices, models, samples, colors):
        cis.append(calc_percentiles(sample, confidence_interval))
        ci = cis[-1][..., model.predict_idx]
        plt.fill_between(all_year[min(idx):][model.predict_idx], ci[0], ci[1],
                         label='Bayesian Estimator at {:.1f} Years Observed Data Span at 90% CI'.format(span), color=color, alpha=0.5)
        one_year_mean_ci.append((ci[1]-ci[0])[:12].mean())
        five_year_mean_ci.append((ci[1]-ci[0]).mean())
        median.append(calc_percentiles(sample, [0.5])[0, model.predict_idx])

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Bayesian Predictions at Various Observed Data Spans')
    plt.legend(loc='lower left')
    plt.grid()

    output_file_name = plots_dir + '/bayesian_example_span.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    plt.figure()
    plt.plot(spans, one_year_mean_ci, 'bo-', label='One Year Mean 90% CI')
    plt.plot(spans, five_year_mean_ci, 'ro-', label='Five Year Mean 90% CI')
    plt.xlabel('Observed Data Span [Years]')
    plt.ylabel('Mean 90% CI')
    plt.title('Bayesian Estimator 90% CI vs Observed Data Span')
    plt.legend(loc='upper right')
    plt.grid()

    output_file_name = plots_dir + '/bayesian_example_span_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    ####################################
    # Show predictions of missing data #
    ####################################
    missing_ratios = np.linspace(0, 1, 5)
    missing_models = []
    missing_guides = []
    missing_samples = []
    for missing_ratio in missing_ratios:
        start_predict = np.floor(missing_ratio * (len(observations) - num_predictions))
        obs_idx = [idx for idx in range(len(observations)) if idx < start_predict or idx >= start_predict + num_predictions]
        missing_models.append(create_model(obs_idx, len(observations) - max(obs_idx) - 1))
        missing_guides.append(fit(missing_models[-1], observations[obs_idx]))
        missing_samples.append(Predictive(missing_models[-1].predict,
                                          guide=missing_guides[-1],
                                          num_samples=num_samples,
                                          return_sites=("_RETURN",))(observations[obs_idx])['_RETURN'])

    # Calculate conditional probabilities of observing the missing samples given the known samples
    missing_prob_observations = []
    missing_prob_all = []
    prob_missing_given_obs = []
    for missing_model, missing_guide in zip(missing_models, missing_guides):
        missing_prob_observations.append(Importance(missing_model, guide=missing_guide, num_samples=num_samples))
        missing_prob_observations[-1].run(observations[missing_model.obs_idx], missing_model.obs_idx)
        missing_prob_all.append(Importance(missing_model, guide=missing_guide, num_samples=num_samples))
        missing_prob_all[-1].run(observations, [*range(len(observations))])
        prob_missing_given_obs.append(missing_prob_all[-1].obs_log_prob() - missing_prob_observations[-1].obs_log_prob())

    # Calculate confidence intervals of predictions
    missing_cis = [calc_percentiles(s[...,m.predict_idx], confidence_interval) for s, m in zip(missing_samples, missing_models)]
    missing_mean_cis = [(ci[1] - ci[0]).mean() for ci in missing_cis]

    # Plot predictions and actuals
    plt.figure()
    for n, (missing_model, missing_ci) in enumerate(zip(missing_models, missing_cis)):
        plt.subplot(len(missing_ratios), 1, n+1)
        plt.fill_between(year[missing_model.predict_idx],
                         missing_ci[0], missing_ci[1], color='r', alpha=0.5)
        plt.plot(year, observations, 'b')
        plt.grid()
        plt.ylabel('Value')
        if n == 0:
            plt.title('Predicting Arbitrary Missing Samples at 90% CI')
        if n < (len(missing_ratios) - 1):
            plt.gca().xaxis.set_tick_params(labelbottom=False)
    
    plt.xlabel('Year')

    output_file_name = plots_dir + '/bayesian_example_missing.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    # Plot mean confidence interval and observation probability versus missing samples location 
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(100 * missing_ratios, missing_mean_cis, 'bo-')
    plt.xlabel('Amount of Data Before First Missing Sample [%]')
    plt.ylabel('Mean 90% CI')
    plt.title('Bayesian Estimator 90% CI vs Missing Samples Location')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(100 * missing_ratios, prob_missing_given_obs, 'ro-')
    plt.xlabel('Amount of Data Before First Missing Sample [%]')
    plt.ylabel('Log of Probability Density')
    plt.title('Missing Samples Probability Density vs Missing Samples Location')
    plt.grid()
    plt.tight_layout()

    output_file_name = plots_dir + '/bayesian_example_missing_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
