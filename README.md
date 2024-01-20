# torch_arima

ARIMA time series implementation in PyTorch implemented as a torch `torch.nn.Module` called `ARIMA`.

## Installation

For local package installation that enables modifying the package source code in place without reinstalling run

```
pip install -e .
```

## Tests

```
python -m ARIMA
```

## Examples

- Examples are based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.
- Raw data downloaded from https://www.key2stats.com/data-set/view/764.

### Maximum Likelihood Estimator

Utilizes `torch` optimizers in order find the maximum likelihood estimator. Run by executing

```
python -m ARIMA.examples.mle
```

The below graphs will be created.

![](/ARIMA/examples/plots/mle_example.png)
![](/ARIMA/examples/plots/mle_example_ratio.png)
![](/ARIMA/examples/plots/mle_example_ratio_ci.png)

### Bayesian Estimator

Utilizes `pyro` which is based on `torch` in order to find the Bayesian posterior. Run by executing

```
python -m ARIMA.examples.bayesian
```

The below graphs will be created.

![](/ARIMA/examples/plots/bayesian_example.png)
![](/ARIMA/examples/plots/bayesian_example_ratio.png)
![](/ARIMA/examples/plots/bayesian_example_ratio_ci.png)

The Bayesian estimator can also estimate missing samples that occur at arbitrary times.

![](/ARIMA/examples/plots/bayesian_example_missing.png)
![](/ARIMA/examples/plots/bayesian_example_missing_ci.png)
