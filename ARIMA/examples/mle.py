'''
Time series Maximum Likelihood Estimator (MLE) forecasting example in PyTorch.

Based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.

Raw data downloaded from https://www.key2stats.com/data-set/view/764.
'''
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
from ARIMA import ARIMA
from ARIMA.examples.utils import load_data, plots_dir, timeit
from ARIMA.examples import __name__ as __examples__name__
from torch.distributions.transforms import ExpTransform
from pyro.ops.stats import quantile

def create_model():
    return ARIMA(3, 0, 1, 0, 1, 2, 12, output_transforms=[ExpTransform()])

@timeit
def fit(model, observations,
        lr_sequence=[(0.005, 100),
                     (0.010, 100)] * 5 +
                    [(0.005, 100),
                     (0.001, 1000)]):
    # Optimize model parameters in order find the Maximum Likelihood Estimator (MLE).
    # In case of an ARIMA model maximizing the likelihood of the observations is equivalent to maxmizing the likelihood of the innovations as the Jacobian determinant of the transformation between the two is equal to one.
    # See a discussion with ChatGPT on the subject at https://chat.openai.com/share/55d34600-6b9d-49ea-b7de-0b70b0e2382f.
    for lr, num_iterations in lr_sequence:
        optimizer = pt.optim.Adam(model.parameters(), lr=lr)
        for count in range(num_iterations):
            optimizer.zero_grad()
            innovations = model(observations)
            variance = (innovations**2).mean()
            variance.backward()
            optimizer.step()
    return dict(innovations = model(observations))

def predict(model, innovations, num_samples, num_predictions):
    variance = (innovations**2).mean()
    new_innovations = pt.randn(num_samples, num_predictions) * variance.sqrt()
    all_innovations = pt.cat([pt.stack([innovations] * num_samples), new_innovations], dim=-1)
    all_observations = model.predict(all_innovations)
    return all_observations[..., len(innovations):]

if __name__ == "__main__" or __examples__name__ == "__main__":
    ##########################################
    # Fit model to data and show predictions #
    ##########################################
    year, observations = load_data()

    model = create_model()

    fit_result = fit(model, observations)
    
    num_samples = 30000
    num_predictions = 5 * 12
    samples = predict(model, fit_result['innovations'], num_samples, num_predictions)

    confidence_interval = [0.05, 0.95]

    plt.figure()

    # Plot observations
    plt.plot(year, observations, color='b', label='Observations')

    # Plot confidence interval of predictions
    samples_year = max(year) + (np.arange(num_predictions) + 1) * np.diff(year).mean()
    ci = quantile(samples, confidence_interval).detach().numpy()
    plt.fill_between([year[-1]] + list(samples_year),
                     [observations[-1]] + list(ci[0]),
                     [observations[-1]] + list(ci[1]),
                     label='Maximum Likelihood Estimates (MLE) at 90% CI', alpha=0.5, color='r')
    
    plt.title('Monthly corticosteroid drug sales in Australia')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid()

    output_file_name = plots_dir + '/mle_example.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    #########################################################
    # Show effect of amount of training data on predictions #
    #########################################################
    ratios = 0.6 ** np.arange(4)
    models = []
    indices = []
    fit_results = []
    samples = []
    for ratio in ratios:
        n = len(observations)
        indices.append(range(round((1 - ratio)*n), n))
        models.append(create_model())
        fit_results.append(fit(models[-1], observations[indices[-1]]))
        samples.append(predict(models[-1], fit_results[-1]['innovations'], num_samples, num_predictions))

    plt.figure()
    spans = np.array(ratios) * (max(year) - min(year))
    colors = ['r', 'g', 'b', 'y']
    cis = []
    one_year_mean_ci = []
    five_year_mean_ci = []
    median = []
    for span, idx, sample, color in zip(spans, indices, samples, colors):
        cis.append(quantile(sample, confidence_interval).detach().numpy())
        plt.fill_between(samples_year, cis[-1][0], cis[-1][1],
                         label='MLE Estimator at {:.1f} Years Observed Data Span at 90% CI'.format(span), color=color, alpha=0.5)
        one_year_mean_ci.append((cis[-1][1]-cis[-1][0])[:12].mean())
        five_year_mean_ci.append((cis[-1][1]-cis[-1][0]).mean())
        median.append(quantile(sample, [0.5])[0].detach().numpy())
    
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('MLE Predictions at Various Observed Data Spans')
    plt.legend(loc='lower left')
    plt.grid()

    output_file_name = plots_dir + '/mle_example_span.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    plt.figure()
    plt.plot(spans, one_year_mean_ci, 'bo-', label='One Year Mean 90% CI')
    plt.plot(spans, five_year_mean_ci, 'ro-', label='Five Year Mean 90% CI')
    plt.xlabel('Observed Data Span [Years]')
    plt.ylabel('Mean 90% CI')
    plt.title('MLE Mean 90% CI vs Observed Data Span')
    plt.legend(loc='upper left')
    plt.grid()

    output_file_name = plots_dir + '/mle_example_span_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
