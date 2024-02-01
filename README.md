# torch_arima

ARIMA time series implementation in PyTorch supporting the following model types:

| Model Type | Location | Description |
|-|-|-|
| ARIMA | `ARIMA.ARIMA` | `torch.nn.Module` with ARIMA polynomial coefficients as parameters and a forward method that converts observations to innovations and a predict method which converts innovations to observations. |
| VARIMA | `ARIMA.VARIMA` | Same as `ARIMA.ARIMA` with support for vector innovations and vector observations. |
| Bayesian ARIMA | `ARIMA.BayesianARIMA` | `pyro.nn.PyroModule` wrapper around `ARIMA.ARIMA` with support for priors to all polynomial coefficients and innovations distribution parameters. |
| Bayesian VARIMA | `ARIMA.BayesianVARIMA` | Same as `ARIMA.VARIMA` with support for vector innovations and vector observations.|

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

All the examples can be run at once by executing

```
python -m ARIMA.examples
```

This will create additional comparisons between the median predictions and confidence intervals of the MLE and Bayesian estimators.

### ARIMA Examples

- Examples are based on https://otexts.com/fpp2/seasonal-arima.html#example-corticosteroid-drug-sales-in-australia.
- Raw data downloaded from https://www.key2stats.com/data-set/view/764.

#### Maximum Likelihood Estimator

Utilizes `torch` optimizers in order to find the maximum likelihood estimator. Run by executing

```
python -m ARIMA.examples.mle
```

The below graphs will be created.

![](/ARIMA/examples/plots/mle_example.png)
![](/ARIMA/examples/plots/mle_example_span.png)
![](/ARIMA/examples/plots/mle_example_span_ci.png)

#### Bayesian Estimator

Utilizes `pyro` which is based on `torch` in order to find the Bayesian posterior. Run by executing

```
python -m ARIMA.examples.bayesian
```

The below graphs will be created.

![](/ARIMA/examples/plots/bayesian_example.png)
![](/ARIMA/examples/plots/bayesian_example_span.png)
![](/ARIMA/examples/plots/bayesian_example_span_ci.png)

The Bayesian estimator can also estimate missing samples that occur at arbitrary times.

![](/ARIMA/examples/plots/bayesian_example_missing.png)
![](/ARIMA/examples/plots/bayesian_example_missing_ci.png)

#### Comparison Between the Maximum Likelihood and Bayesian Estimators

![](/ARIMA/examples/plots/compare_example_median.png)
![](/ARIMA/examples/plots/compare_example_span_ci.png)

It can be seen that the two estimators have different median predictions, and that as less observed data is available the MLE estimator becomes more confident in its predictions, whereas the Bayesian estimator becomes less confident in its predicitons, especially for the short term predictions.

### Bayesian VARIMA Example

- In this example we predict short term mortality fluctuations in England & Wales.
- The raw data was downlaoded from the [Human Mortality Database](https://www.mortality.org) from https://www.mortality.org/File/GetDocument/Public/STMF/Outputs/GBRTENWstmfout.csv.

The example can be run by executing

```
python -m ARIMA.examples.mortality
```

The below graph shows predicted weekly death counts for males and females. The model captures annual periodic changes in mortality and correlations between female and male death counts.

![](/ARIMA/examples/plots/mortality_example_monthly.png)

Viewed as an yearly moving sum the COVID-19 effect on death counts can be viewed more clearly as annual periodic changes in mortality are averaged out.

![](/ARIMA/examples/plots/mortality_example_yearly.png)

The effect of COVID-19 on short term death count predictions can be visualized by comparing predictions of a model that did not observe death counts during the COVID-19 pandemic (a.k.a. Pre COVID model), to a model that observed the most up to date data available (a.k.a. Post COVID model).

![](/ARIMA/examples/plots/mortality_example_yearly_pre_post_covid.png)

The importance of using a VARIMA model, rather then a model comprised of two independent ARIMA models (a.k.a. Multiple ARIMA model), can be seen in the graph below where the confidence interval of the VARIMA model is much larger (as should be) than that of the Multiple ARIMA model, as it correctly captures the correlation between death counts of females and males.

![](/ARIMA/examples/plots/mortality_example_yearly_total.png)
