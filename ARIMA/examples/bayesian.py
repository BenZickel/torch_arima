'''
Time series Bayesian estimator forecasting example based on PyTorch and Pyro.

Based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.

Raw data downloaded from https://www.key2stats.com/data-set/view/764.
'''
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pyro
from pyro.infer import WeighedPredictive, MHResampler
from pyro.ops.stats import quantile, energy_score_empirical
from ARIMA import BayesianARIMA
from ARIMA.examples.cross_validation import cross_validation_folds, score_fold
from ARIMA.examples.utils import load_data, plots_dir, timeit, moving_sum
from ARIMA.examples import __name__ as __examples__name__
from torch.distributions.transforms import ExpTransform, AffineTransform

def create_model(obs_idx, num_predictions, observations, model_args=(3, 0, 1, 0, 1, 2, 12)):
    # Create model with non-overlapping observed and predicted sample indices.
    predict_idx = [*range(max(obs_idx) + 1 + num_predictions)]
    predict_idx = [idx for idx in predict_idx if idx not in obs_idx]
    # Normalize observations by an output transform
    mean_log = observations.log().mean()
    std_log = observations.log().std()
    output_transforms = [AffineTransform(loc=mean_log, scale=std_log), ExpTransform()]
    return BayesianARIMA(*model_args,
                         obs_idx=obs_idx, predict_idx=predict_idx,
                         output_transforms=output_transforms)

@timeit
def fit(model,
        lr_sequence=[(0.005, 100),
                     (0.010, 100)] * 5 +
                    [(0.005, 100),
                     (0.001, 100)],
        loss=pyro.infer.JitTrace_ELBO,
        loss_params=dict(num_particles=20, vectorize_particles=True, ignore_jit_warnings=True)):
    # Create posterior for Bayesian model
    guide = pyro.infer.autoguide.guides.AutoMultivariateNormal(model)
    guide()
    guide.loc.data[:] = 0
    guide.scale_unconstrained.data[:] = -5
    loss = loss(**loss_params)
    for lr, num_iter in lr_sequence:
        optimizer = pyro.optim.Adam(dict(lr=lr))
        svi = pyro.infer.SVI(model, guide, optimizer, loss=loss)
        for count in range(num_iter):
            svi.step()
    return guide

if __name__ == '__main__' or __examples__name__ == '__main__':
    ##########################################
    # Fit model to data and show predictions #
    ##########################################
    year, observations = load_data()

    num_predictions = 5 * 12
    obs_idx = range(len(observations))

    model = create_model(obs_idx, num_predictions, observations)
    conditioned_model = pyro.poutine.condition(model, data={'observations': observations[model.obs_idx]})
    conditioned_predict = pyro.poutine.condition(model.predict, data={'observations': observations[model.obs_idx]})

    guide = fit(conditioned_model)

    # Make predictions
    num_samples = 30000
    predictive = WeighedPredictive(conditioned_predict,
                                   guide=guide,
                                   num_samples=num_samples,
                                   parallel=True,
                                   return_sites=('_RETURN',))
    resampler = MHResampler(predictive)
    while resampler.get_total_transition_count() < num_samples:
        samples = resampler(model_guide=conditioned_model)
        samples = samples.samples['_RETURN']
    confidence_interval = [0.05, 0.95]

    plt.figure()

    # Plot observations
    plt.plot(year[model.obs_idx], observations[model.obs_idx], 'b', label='Observations')

    # Plot confidence interval of predictions
    all_year = np.concatenate((year, year[-1] + (np.arange(len(model.obs_idx) + len(model.predict_idx) - len(year)) + 1) * np.diff(year).mean()))
    idx = sorted(set(np.clip([min(model.predict_idx) - 1, max(model.predict_idx) + 1] + model.predict_idx, 0, len(all_year) - 1)))
    ci = quantile(samples[:,idx], confidence_interval)
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
    ratios = (0.6 ** np.arange(4))[::-1]
    models = []
    indices = []
    guides = []
    samples = []
    for ratio in ratios:
        n = len(observations)
        indices.append(range(round((1 - ratio)*n), n))
        models.append(create_model([*range(len([*indices[-1]]))], num_predictions, observations[indices[-1]]))
        conditioned_model = pyro.poutine.condition(models[-1], data={'observations': observations[indices[-1]]})
        conditioned_predict = pyro.poutine.condition(models[-1].predict, data={'observations': observations[indices[-1]]})
        guides.append(fit(conditioned_model))
        predictive = WeighedPredictive(conditioned_predict,
                                       guide=guides[-1],
                                       num_samples=num_samples,
                                       parallel=True,
                                       return_sites=("_RETURN",))
        resampler = MHResampler(predictive)
        while resampler.get_total_transition_count() < num_samples:
            sample = resampler(model_guide=conditioned_model).samples['_RETURN']
        samples.append(sample)

    plt.figure()
    spans = np.array(ratios) * (max(year) - min(year))
    colors = ['r', 'g', 'b', 'y'][::-1]
    cis = []
    one_year_mean_ci = []
    five_year_mean_ci = []
    moving_sum_median = []
    for span, idx, model, sample, color in zip(spans, indices, models, samples, colors):
        cis.append(quantile(sample, confidence_interval))
        ci = cis[-1][..., model.predict_idx]
        plt.fill_between(all_year[min(idx):][model.predict_idx], ci[0], ci[1],
                         label='Bayesian Estimator at {:.1f} Years Observed Data Span at 90% CI'.format(span), color=color, alpha=0.5)
        one_year_mean_ci.append((ci[1]-ci[0])[:12].mean())
        five_year_mean_ci.append((ci[1]-ci[0]).mean())
        moving_sum_median.append(quantile(moving_sum(sample[..., model.predict_idx], 12, 1), [0.5])[0])

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
    plt.legend(loc='upper left')
    plt.grid()

    output_file_name = plots_dir + '/bayesian_example_span_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    ####################################
    # Show predictions of missing data #
    ####################################
    def posterior_predictive_sampler(obs, train_idx, test_idx, *args, **kwargs):
        model = create_model(train_idx, max(max(test_idx) - max(train_idx), 0), obs[train_idx], *args, **kwargs)
        conditioned_model = pyro.poutine.condition(model, data={'observations': obs[train_idx]})
        conditioned_predict = pyro.poutine.condition(model.predict, data={'observations': obs[train_idx]})
        guide = fit(conditioned_model)
        predictive = WeighedPredictive(conditioned_predict,
                                       guide=guide,
                                       num_samples=num_samples,
                                       parallel=True,
                                       return_sites=("_RETURN",))
        resampler = MHResampler(predictive)
        while resampler.get_total_transition_count() < num_samples:
            posterior_predictive_samples = resampler(model_guide=conditioned_model).samples['_RETURN']
        return posterior_predictive_samples[..., test_idx]

    num_folds = 5
    missing_samples = []
    missing_energy_score = []
    missing_idx = []
    for obs, train_idx, test_idx, start_idx in cross_validation_folds(observations, num_predictions, num_folds):
        score, posterior_predictive_samples = score_fold(posterior_predictive_sampler, obs, train_idx, test_idx)
        missing_samples.append(posterior_predictive_samples)
        missing_energy_score.append(score / np.sqrt(len(test_idx)))
        missing_idx.append(test_idx)

    # Calculate confidence intervals of predictions
    missing_cis = [quantile(s, confidence_interval) for s in missing_samples]
    missing_mean_cis = [(ci[1] - ci[0]).mean() for ci in missing_cis]

    # Plot predictions and actuals
    plt.figure()
    for n, missing_ci in enumerate(missing_cis):
        plt.subplot(num_folds, 1, n+1)
        plt.fill_between(year[missing_idx[n]],
                         missing_ci[0], missing_ci[1], color='r', alpha=0.5)
        plt.plot(year, observations, 'b')
        plt.grid()
        plt.ylabel('Value')
        if n == 0:
            plt.title('Predicting Arbitrary Missing Samples at 90% CI')
        if n < (num_folds - 1):
            plt.gca().xaxis.set_tick_params(labelbottom=False)
    
    plt.xlabel('Year')

    output_file_name = plots_dir + '/bayesian_example_missing.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    # Plot mean confidence interval and observation probability versus missing samples location 
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(100 * np.linspace(0, 1, num_folds), missing_mean_cis, 'bo-')
    plt.xlabel('Amount of Data Before First Missing Sample [%]')
    plt.ylabel('Mean 90% CI')
    plt.title('Bayesian Estimator 90% CI vs Missing Samples Location')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(100 * np.linspace(0, 1, num_folds), missing_energy_score, 'ro-')
    plt.xlabel('Amount of Data Before First Missing Sample [%]')
    plt.ylabel('Per Sample Energy Score')
    plt.title('Missing Samples Energy Score vs Missing Samples Location')
    plt.grid()
    plt.tight_layout()

    output_file_name = plots_dir + '/bayesian_example_missing_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
