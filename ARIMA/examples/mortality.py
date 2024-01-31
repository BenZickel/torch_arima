'''
Multivariate time series Bayesian estimator forecasting example based on PyTorch and Pyro.
'''
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pyro
from torch.distributions.transforms import ExpTransform, AffineTransform
from pyro.infer import Predictive
from ARIMA import BayesianVARIMA
from ARIMA.Innovations import NormalInnovationsVector, MultivariateNormalInnovations
from ARIMA.examples.utils import load_mortality_data, calc_percentiles, moving_sum, plots_dir
from ARIMA.examples import __name__ as __examples__name__

def create_model(obs_idx, n, num_predictions, observations, innovations=MultivariateNormalInnovations):
    # Create model with non-overlapping observed and predicted sample indices.
    predict_idx = [*range(max(obs_idx) + 1 + num_predictions)]
    predict_idx = [idx for idx in predict_idx if idx not in obs_idx]
    # Normalize observations by an output transform
    mean_log = observations.log().mean()
    std_log = observations.log().std()
    output_transforms = [AffineTransform(loc=mean_log, scale=std_log), ExpTransform()]
    return BayesianVARIMA(2, 0, 1, 0, 1, 1, 12, drift=True, n=n,
                          obs_idx=obs_idx, predict_idx=predict_idx,
                          output_transforms=output_transforms, innovations=innovations)

def fit(model, observations,
        lr_sequence=[(0.005, 200),
                     (0.010, 200)] * 5 +
                    [(0.005, 200),
                     (0.001, 200)],
        loss=pyro.infer.Trace_ELBO,
        loss_params=dict(num_particles=100, vectorize_particles=True)):
    # Create posterior for Bayesian model
    guide = pyro.infer.autoguide.guides.AutoMultivariateNormal(model)
    guide(observations)
    guide.loc.data[:] = 0
    for lr, num_iter in lr_sequence:
        optimizer = pyro.optim.Adam(dict(lr=lr))
        svi = pyro.infer.SVI(model, guide, optimizer, loss=loss(**loss_params))
        for count in range(num_iter):
            svi.step(observations)
    return guide

if __name__ == "__main__" or __examples__name__ == "__main__":
    ##########################################
    # Fit model to data and show predictions #
    ##########################################
    data = load_mortality_data()
    observations = pt.Tensor(data.values)
    year = np.array([d.year + (d.month - 1) / 12 for d in data.index])

    num_predictions = 5 * 12
    obs_idx = range(len(observations))

    pyro.clear_param_store()

    model = create_model(obs_idx, observations.shape[-1], num_predictions, observations)

    guide = fit(model, observations[model.obs_idx])

    # Make predictions
    num_samples = 1000
    predictive = Predictive(model,
                            guide=guide,
                            num_samples=num_samples,
                            return_sites=("_RETURN",))
    samples = predictive()['_RETURN']

    confidence_interval = [0.05, 0.95]

    # Plot monthly death counts
    plt.figure()

    colors=['r', 'b']
    for n, color, name in zip(range(observations.shape[-1]), colors, data.columns):
        # Plot observations
        plt.plot(year[model.obs_idx], observations[model.obs_idx][:,n], color, label='Observed ' + name)

        # Plot confidence interval of predictions
        all_year = np.concatenate((year, year[-1] + (np.arange(len(model.obs_idx) + len(model.predict_idx) - len(year)) + 1) * np.diff(year).mean()))
        idx = sorted(set(np.clip([min(model.predict_idx) - 1, max(model.predict_idx) + 1] + model.predict_idx, 0, len(all_year) - 1)))
        ci = calc_percentiles(samples[:,idx,n], confidence_interval)
        plt.fill_between(all_year[idx], ci[0], ci[1], color=color, alpha=0.5, label='Predicted {} at 90% CI'.format(name))
        
    plt.xlabel('Year')
    plt.ylabel('Monthly Death Count')
    plt.title('England & Wales Monthly Death Count over Time')
    plt.legend(loc='upper left')
    plt.grid()

    output_file_name = plots_dir + '/mortality_example_monthly.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    # Plot yearly death counts
    window_len = 12
    period_samples = moving_sum(samples, window_len, 1)
    period_all_year = all_year[(window_len-1):]

    plt.figure()

    colors=['r', 'b']
    for n, color, name in zip(range(period_samples.shape[-1]), colors, data.columns):
        # Plot observations
        median = calc_percentiles(period_samples, [0.5])[0]
        plt.plot(period_all_year,
                 median[:, n], color, label='Observed ' + name)

        # Plot confidence interval of predictions
        ci = calc_percentiles(period_samples[:,-(num_predictions+1):,n], confidence_interval)
        plt.fill_between(period_all_year[-(num_predictions+1):], ci[0], ci[1], color=color, alpha=0.5, label='Predicted {} at 90% CI'.format(name))
        
    plt.xlabel('Year')
    plt.ylabel('Yearly Death Count')
    plt.title('England & Wales Yearly Death Count over Time')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()

    output_file_name = plots_dir + '/mortality_example_yearly.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    ##########################################
    # Compare to pre COVID model predictions #
    ##########################################
    pre_covid_observations = observations[year < 2020]
    pre_covid_year = year[year < 2020]

    pre_covid_num_predictions = len(observations) + num_predictions - len(pre_covid_observations)
    pre_covid_obs_idx = range(len(pre_covid_observations))

    pyro.clear_param_store()

    pre_covid_model = create_model(pre_covid_obs_idx, pre_covid_observations.shape[-1], pre_covid_num_predictions, pre_covid_observations)

    pre_covid_guide = fit(pre_covid_model, pre_covid_observations[pre_covid_model.obs_idx])

    # Make predictions
    pre_covid_predictive = Predictive(pre_covid_model,
                                      guide=pre_covid_guide,
                                      num_samples=num_samples,
                                      return_sites=("_RETURN",))
    pre_covid_samples = pre_covid_predictive()['_RETURN']

    # Plot yearly death counts
    window_len = 12
    pre_covid_period_samples = moving_sum(pre_covid_samples, window_len, 1)
    pre_covid_period_all_year = all_year[(window_len-1):]

    plt.figure()

    # Post COVID predictions
    colors=['r', 'b']
    for n, color, name in zip(range(period_samples.shape[-1]), colors, data.columns):
        # Plot observations
        median = calc_percentiles(period_samples, [0.5])[0]
        plt.plot(period_all_year,
                 median[:, n], color, linewidth=3)

        # Plot confidence interval of predictions
        ci = calc_percentiles(period_samples[:,-(num_predictions+1):,n], confidence_interval)
        plt.fill_between(period_all_year[-(num_predictions+1):], ci[0], ci[1], color=color, alpha=0.6, label='Predicted Post COVID {} at 90% CI'.format(name))
    
    # Pre COVID predictions
    colors=['g', 'y']
    for n, color, name in zip(range(pre_covid_period_samples.shape[-1]), colors, data.columns):
        # Plot observations
        median = calc_percentiles(pre_covid_period_samples, [0.5])[0]
        plt.plot(pre_covid_period_all_year,
                 median[:, n], color)

        # Plot confidence interval of predictions
        ci = calc_percentiles(pre_covid_period_samples[:,-(pre_covid_num_predictions+1):,n], confidence_interval)
        plt.fill_between(period_all_year[-(pre_covid_num_predictions+1):], ci[0], ci[1], color=color, alpha=0.6, label='Predicted Pre COVID {} at 90% CI'.format(name))
        
    plt.xlabel('Year')
    plt.ylabel('Yearly Death Count')
    plt.title('England & Wales Yearly Death Count over Time')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()

    output_file_name = plots_dir + '/mortality_example_yearly_pre_post_covid.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    #######################################
    # Compare VARIMA to ARIMA predictions #
    #######################################
    pyro.clear_param_store()

    multi_arima_model = create_model(obs_idx, observations.shape[-1], num_predictions, observations, innovations=NormalInnovationsVector)

    multi_arima_guide = fit(multi_arima_model, observations[multi_arima_model.obs_idx])

    # Make predictions
    predictive = Predictive(multi_arima_model,
                            guide=multi_arima_guide,
                            num_samples=num_samples,
                            return_sites=("_RETURN",))
    multi_arima_samples = predictive()['_RETURN']

    # Calculate yearly moving sum
    period_multi_arima_samples = moving_sum(multi_arima_samples, window_len, 1)
    
    # Plot sum over males and females
    plt.figure()

    colors = ['r', 'b']
    linewidths = [3, 1]
    for sample, color, linewidth, name in zip([period_samples.sum(2),
                               period_multi_arima_samples.sum(2)],
                              colors, linewidths,
                              ['VARIMA', 'Multiple ARIMA']):
        # Plot confidence interval of predictions
        ci = calc_percentiles(sample[:,-(num_predictions+1):], confidence_interval)
        plt.fill_between(period_all_year[-(num_predictions+1):], ci[0], ci[1], color=color, alpha=0.5, label='{} Predictions at 90% CI'.format(name))
        # Plot observations and median
        median = calc_percentiles(sample, [0.5])[0]
        plt.plot(period_all_year, median, color, linewidth=linewidth, label='{} Predictions Median'.format(name))
    
    plt.xlabel('Year')
    plt.ylabel('Total (Males and Females) Yearly Death Count')
    plt.title('England & Wales Total (Males and Females) Yearly Death Count')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()

    output_file_name = plots_dir + '/mortality_example_yearly_total.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
