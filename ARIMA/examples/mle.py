'''
Time series Maximum Likelihood Estimator (MLE) forecasting example in PyTorch.

Based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.

Raw data downloaded from https://www.key2stats.com/data-set/view/764.
'''
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import os
from ARIMA import ARIMA
from ARIMA.examples.utils import load_data, calc_percentiles, plot

def fit(model, observations,
        lr_sequence=[(0.005, 100),
                     (0.010, 100)] * 5 +
                    [(0.005, 100),
                     (0.001, 500)]):
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
    samples = []
    for count in range(num_samples):
        new_innovations = pt.randn(num_predictions) * variance.sqrt()
        all_innovations = pt.cat([innovations, new_innovations])
        all_observations = model.predict(all_innovations).detach().numpy()
        samples.append(all_observations)
    return np.array(samples)

if __name__ == "__main__":
    ##########################################
    # Fit model to data and show predictions #
    ##########################################
    year, observations = load_data()

    model = ARIMA(3,0,1,0,1,2,12)

    fit_result = fit(model, observations)
    
    num_samples = 1000
    num_predictions = 5 * 12
    samples = predict(model, fit_result['innovations'], num_samples, num_predictions)

    plt.figure()
    plot(year, calc_percentiles(samples, [0.05, 0.95]), observations,
         observations_label='Observations',
         samples_label='Maximum Likelihood Estimates (MLE) at 90% CI')
    plt.title('Monthly corticosteroid drug sales in Australia')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid()
    
    plots_dir = os.path.dirname(__file__) + '/plots'
    os.makedirs(plots_dir, exist_ok=True)

    output_file_name = plots_dir + '/mle_example.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    #########################################################
    # Show effect of amount of training data on predictions #
    #########################################################
    ratios = [1,1/2,1/4,1/8]
    models = []
    indices = []
    fit_results = []
    samples = []
    for ratio in ratios:
        n = len(observations)
        indices.append(range(round((1 - ratio)*n), n))
        models.append(ARIMA(3,0,1,0,1,2,12))
        fit_results.append(fit(models[-1], observations[indices[-1]]))
        samples.append(predict(models[-1], fit_results[-1]['innovations'], num_samples, num_predictions))

    plt.figure()
    colors = ['r', 'g', 'b', 'y']
    cis = []
    one_year_mean_ci = []
    five_year_mean_ci = []
    for ratio, idx, sample, color in zip(ratios, indices, samples, colors):
        cis.append(calc_percentiles(sample, [0.05, 0.95]))
        plot(year[idx], cis[-1], observations[idx],
             samples_label='MLE {}% of data at 90% CI'.format(100 * ratio), samples_color=color)
        ci = cis[-1][:,-num_predictions:]
        one_year_mean_ci.append((ci[1]-ci[0])[:12].mean())
        five_year_mean_ci.append((ci[1]-ci[0]).mean())

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('MLE Predictions at Various Ratios of Observed Data')
    plt.legend(loc='lower left')
    plt.grid()

    output_file_name = plots_dir + '/mle_example_ratio.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)

    plt.figure()
    plt.plot(ratios, one_year_mean_ci, 'bo-', label='One Year Mean 90% CI')
    plt.plot(ratios, five_year_mean_ci, 'ro-', label='Five Year Mean 90% CI')
    plt.xlabel('Ratio of Observed Data')
    plt.ylabel('Mean 90% CI')
    plt.title('MLE Mean 90% CI vs Ratio of Observed Data')
    plt.legend(loc='upper left')
    plt.grid()

    output_file_name = plots_dir + '/mle_example_ratio_ci.png'
    plt.savefig(output_file_name)
    print('Saved output file ' + output_file_name)
