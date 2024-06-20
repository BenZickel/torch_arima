import ARIMA.examples
ARIMA.examples.__name__ = __name__

from ARIMA.examples.utils import plots_dir
from matplotlib import pyplot as plt
import numpy as np

# Run all examples
import ARIMA.examples.mle
import ARIMA.examples.bayesian
import ARIMA.examples.mortality

plt.figure()
for name, plot_spec, example in [('MLE', 'o-', ARIMA.examples.mle),
                                 ('Bayesian', 'x-', ARIMA.examples.bayesian)]:
    spans = np.array(example.ratios) * (max(example.year) - min(example.year))
    plt.plot(spans, example.one_year_mean_ci, 'b' + plot_spec, label='{} Estimator One Year Mean 90% CI'.format(name))
    plt.plot(spans, example.five_year_mean_ci, 'r' + plot_spec, label='{} Estimator Five Year Mean 90% CI'.format(name))
plt.xlabel('Observed Data Span [Years]')
plt.ylabel('Mean 90% CI')
plt.title('Estimator 90% CI vs Observed Data Span')
plt.legend(loc='upper right')
plt.grid()

output_file_name = plots_dir + '/compare_example_span_ci.png'
plt.savefig(output_file_name)
print('Saved output file ' + output_file_name)

plt.figure()
colors = ['r', 'g', 'b', 'y']
for name, plot_spec, example in [('MLE', '--', ARIMA.examples.mle),
                                 ('Bayesian', '-', ARIMA.examples.bayesian)]:
    spans = np.array(example.ratios) * (max(example.year) - min(example.year))
    for plot_num, (span, median, color) in enumerate(zip(spans, example.moving_sum_median, colors)):
        plt.subplot(2, 2, plot_num + 1)
        plt.plot(range(1, len(median) + 1), median / 12, color + plot_spec, label=name)
        plt.title('Span = {:.1f} Years'.format(span))
        if plot_num in [2, 3]:
            plt.xlabel('Prediction Month')
        if plot_num in [0, 2]:
            plt.ylabel('Prediction Median')
        plt.legend(loc='upper right')
        plt.grid(True)
plt.suptitle('Per Estimator Previous Year Prediction Median')
plt.tight_layout()

output_file_name = plots_dir + '/compare_example_median.png'
plt.savefig(output_file_name)
print('Saved output file ' + output_file_name)
