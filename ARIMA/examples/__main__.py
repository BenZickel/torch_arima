import ARIMA.examples
ARIMA.examples.__name__ = __name__

from ARIMA.examples.utils import plots_dir
from matplotlib import pyplot as plt
import numpy as np

# Run all examples
import ARIMA.examples.mle
import ARIMA.examples.bayesian

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
