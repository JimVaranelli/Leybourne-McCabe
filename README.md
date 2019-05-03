# Leybourne-McCabe
Python implementation of Leybourne-McCabe (1994, 1999) stationarity test.

## Parameters
x : array_like, 1d \
&nbsp;&nbsp;&nbsp;&nbsp;data series \
arlags : int \
&nbsp;&nbsp;&nbsp;&nbsp;number of autoregressive terms to include, default=None \
regression : {'c','ct'} \
&nbsp;&nbsp;&nbsp;&nbsp;Constant and trend order to include in regression \
&nbsp;&nbsp;&nbsp;&nbsp;* 'c'  : constant only (default) \
&nbsp;&nbsp;&nbsp;&nbsp;* 'ct' : constant and trend \
method : {'mle','ols'} \
&nbsp;&nbsp;&nbsp;&nbsp;Method used to estimate ARIMA(p, 1, 1) filter model \
&nbsp;&nbsp;&nbsp;&nbsp;* 'mle' : condition sum of squares maximum likelihood (default) \
&nbsp;&nbsp;&nbsp;&nbsp;* 'ols' : two-stage least squares \
varest : {'var94','var99'} \
&nbsp;&nbsp;&nbsp;&nbsp;Method used for residual variance estimation \
&nbsp;&nbsp;&nbsp;&nbsp;* 'var94' : method used in original Leybourne-McCabe paper (1994) (default) \
&nbsp;&nbsp;&nbsp;&nbsp;* 'var99' : method used in follow-up paper (1999)

## Returns
lmstat : float \
&nbsp;&nbsp;&nbsp;&nbsp;test statistic \
pvalue : float \
&nbsp;&nbsp;&nbsp;&nbsp;based on MC-derived critical values \
arlags : int \
&nbsp;&nbsp;&nbsp;&nbsp;AR(p) order used to create the filtered series \
cvdict : dict \
&nbsp;&nbsp;&nbsp;&nbsp;critical values for the test statistic at the 1%, 5%, and 10% levels

## Notes
Critical values for the two different models are generated through Monte Carlo simulation using 1,000,000 replications and 2000 data points

H0 = series is stationary

Basic process is to create a filtered series which removes the AR(p) effects from the series under test followed by an auxiliary regression similar to that of Kwiatkowski et al (1992). The AR(p) coefficients are obtained by estimating an ARIMA(p, 1, 1) model. Two methods are provided for ARIMA estimation: MLE and two-stage least squares. Two methods are provided for residual variance estimation used in the calculation of the test statistic. The first method ('var94') is the mean of the squared residuals from the filtered regression. The second method ('var99') is the MA(1) coefficient times the mean of the squared residuals from the ARIMA(p, 1, 1) filtering model. An empirical autolag procedure is provided. In this context, the number of lags is equal to the number of AR(p) terms used in the filtering step. The number of AR(p) terms is set equal to the to the first PACF falling within the 95% confidence interval. Maximum nuber of AR lags is limited to 1/2 series length.

## References
Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54: 159–178.

Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a unit root. Journal of Business and Economic Statistics, 12: 157–166.

Leybourne, S.J., & McCabe, B.P.M. (1999). Modified stationarity tests with data-dependent model-selection rules. Journal of Business and Economic Statistics, 17: 264-270.

Schwert, G W. (1987). Effects of model specification on tests for unit roots in macroeconomic data. Journal of Monetary Economics, 20: 73–103.

## Requirements
Python 3.6 \
Numpy 1.13.1 \
Statsmodels 0.9.0 \
Pandas 0.20.3

## Running
There are no parameters. The program is set up to access test files in the .\results directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced output for the test file.
