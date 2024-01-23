import doctest
import ARIMA.TimeSeries
import ARIMA.BayesianTimeSeries

doctest.testmod(m=ARIMA.TimeSeries)
doctest.testmod(m=ARIMA.BayesianTimeSeries)