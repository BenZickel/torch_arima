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

<a name="mle_example"></a>
The below graphs will be created.

![](/ARIMA/examples/plots/mle_example.png)
<a name="mle_example_span"></a>
![](/ARIMA/examples/plots/mle_example_span.png)
<a name="mle_example_span_ci"></a>
![](/ARIMA/examples/plots/mle_example_span_ci.png)

#### Bayesian Estimator

Utilizes `pyro` which is based on `torch` in order to find the Bayesian posterior. Run by executing

```
python -m ARIMA.examples.bayesian
```

<a name="bayesian_example"></a>
The below graphs will be created.

![](/ARIMA/examples/plots/bayesian_example.png)
<a name="bayesian_example_span"></a>
![](/ARIMA/examples/plots/bayesian_example_span.png)
<a name="bayesian_example_span_ci"></a>
![](/ARIMA/examples/plots/bayesian_example_span_ci.png)

<a name="bayesian_example_missing"></a>
The Bayesian estimator can also estimate missing samples that occur at arbitrary times.

![](/ARIMA/examples/plots/bayesian_example_missing.png)
<a name="bayesian_example_missing_ci"></a>
![](/ARIMA/examples/plots/bayesian_example_missing_ci.png)

#### Comparison Between the Maximum Likelihood and Bayesian Estimators

<a name="compare_example_median"></a>
![](/ARIMA/examples/plots/compare_example_median.png)
<a name="compare_example_span_ci"></a>
![](/ARIMA/examples/plots/compare_example_span_ci.png)

It can be seen that the two estimators have different median predictions, and that as less observed data is available the MLE estimator becomes more confident in its predictions, whereas the Bayesian estimator becomes less confident in its predicitons, especially for the short term predictions.

### Bayesian VARIMA Example

- In this example we predict short term mortality fluctuations in England & Wales.
- The raw data was downlaoded from the [Human Mortality Database](https://www.mortality.org) from https://www.mortality.org/File/GetDocument/Public/STMF/Outputs/GBRTENWstmfout.csv.

The example can be run by executing

```
python -m ARIMA.examples.mortality
```

<a name="mortality_example_monthly"></a>
The below graph shows predicted weekly death counts for males and females. The model captures annual periodic changes in mortality and correlations between female and male death counts.

![](/ARIMA/examples/plots/mortality_example_monthly.png)

<a name="mortality_example_yearly"></a>
Viewed as an yearly moving sum the COVID-19 effect on death counts can be viewed more clearly as annual periodic changes in mortality are averaged out.

![](/ARIMA/examples/plots/mortality_example_yearly.png)

<a name="mortality_example_yearly_pre_post_covid"></a>
The effect of COVID-19 on short term death count predictions can be visualized by comparing predictions of a model that did not observe death counts during the COVID-19 pandemic (a.k.a. Pre COVID model), to a model that observed the most up to date data available (a.k.a. Post COVID model).

![](/ARIMA/examples/plots/mortality_example_yearly_pre_post_covid.png)

<a name="mortality_example_yearly_total"></a>
The importance of using a VARIMA model, rather then a model comprised of two independent ARIMA models (a.k.a. Multiple ARIMA model), can be seen in the graph below where the confidence interval of the VARIMA model is much larger (as should be) than that of the Multiple ARIMA model, as it correctly captures the correlation between death counts of females and males.

![](/ARIMA/examples/plots/mortality_example_yearly_total.png)

## Design

An ARIMA(p,d,q) time series is defined by the equation (courtesy of [Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average))

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/12cc5e99bc1595494ef8219d70b304784e3933d0">

with $X_i$ being the observations, $\epsilon_i$ being the innovations, and $L$ is the [lag operator](https://en.wikipedia.org/wiki/Lag_operator).

The determinant of the Jacobian of the transformation from innovations to observations is equal to one since

```math
\begin{align}
\frac{\partial X_i}{\partial \epsilon_i} &= 1 \text{  for all  } i \\
\frac{\partial X_i}{\partial \epsilon_j} &= 0 \text{  for all  } j > i
\end{align}
```

This means that the ARIMA transformation can be viewed as a change of random variable from innovations to observations, in which the probability density of innovations is equal to the probability density of the observations, which is how the core of the ARIMA module is implemented in [Transform.py](/ARIMA/Transform.py).
