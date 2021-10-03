import sys
import os
import time
import numpy as np
import pandas as pd
from builtins import int
# statsmodels 0.13 deprecates arima_model.ARIMA
# in favor of arima.model.ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.tsatools import lagmat
from numpy.testing import assert_equal, assert_almost_equal

class Leybourne(object):
    """
    Class wrapper for Leybourne-Mccabe stationarity test
    """
    def __init__(self):
        """
        Asymptotic critical values for the two different models specified
        for the Leybourne-McCabe stationarity test. Asymptotic CVs are the
        same as the asymptotic CVs for the KPSS stationarity test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        1,000,000 replications and 2000 data points.
        """
        self.__leybourne_critical_values = {}
        # constant-only model
        self.__c = ((99.900, 0.0169233), (99.000, 0.0247863), (98.000, 0.0287636),
                    (97.000, 0.0317512), (96.000, 0.0342505), (95.000, 0.0364872),
                    (92.500, 0.0415061), (90.000, 0.0459481), (85.000, 0.0542763),
                    (80.000, 0.0621976), (75.000, 0.0702117), (70.000, 0.0785789),
                    (65.000, 0.0968259), (60.000, 0.0968259), (57.500, 0.101951),
                    (55.000, 0.107248), (52.500, 0.112855), (50.000, 0.118809),
                    (47.500, 0.125104), (45.000, 0.131743), (42.500, 0.138939),
                    (40.000, 0.146608), (37.500, 0.154828), (35.000, 0.163827),
                    (32.500, 0.173569), (30.000, 0.184215), (27.500, 0.196048),
                    (25.000, 0.209452), (22.500, 0.224259), (20.000, 0.24128),
                    (17.500, 0.260842), (15.000, 0.283831), (12.500, 0.311703),
                    (10.000, 0.347373), (7.500, 0.393998), (5.000, 0.46169),
                    (2.500, 0.580372), (1.000, 0.743491), (0.900, 0.763297),
                    (0.800, 0.785173), (0.700, 0.809092), (0.600, 0.83664),
                    (0.500, 0.869455), (0.400, 0.909901), (0.300, 0.962597),
                    (0.200, 1.03998), (0.100, 1.16701), (0.001, 2.84682))
        self.__leybourne_critical_values['c'] = np.asarray(self.__c)
        # constant+trend model
        self.__ct = ((99.900, 0.0126788), (99.000, 0.0172984), (98.000, 0.0194624),
                     (97.000, 0.0210446), (96.000, 0.0223274), (95.000, 0.0234485),
                     (92.500, 0.0258551), (90.000, 0.0279374), (85.000, 0.0315677),
                     (80.000, 0.0349355), (75.000, 0.0381676), (70.000, 0.0413931),
                     (65.000, 0.0446997), (60.000, 0.0481063), (57.500, 0.0498755),
                     (55.000, 0.0517089), (52.500, 0.0536157), (50.000, 0.0555732),
                     (47.500, 0.0576502), (45.000, 0.059805), (42.500, 0.062043),
                     (40.000, 0.064408), (37.500, 0.0669198), (35.000, 0.0696337),
                     (32.500, 0.0725157), (30.000, 0.0756156), (27.500, 0.079006),
                     (25.000, 0.0827421), (22.500, 0.086865), (20.000, 0.09149),
                     (17.500, 0.0967682), (15.000, 0.102787), (12.500, 0.110122),
                     (10.000, 0.119149), (7.500, 0.130935), (5.000, 0.147723),
                     (2.500, 0.177229), (1.000, 0.216605), (0.900, 0.221306),
                     (0.800, 0.226324), (0.700, 0.23257), (0.600, 0.239896),
                     (0.500, 0.248212), (0.400, 0.258809), (0.300, 0.271849),
                     (0.200, 0.29052), (0.100, 0.324278), (0.001, 0.607007))
        self.__leybourne_critical_values['ct'] = np.asarray(self.__ct)

    def __leybourne_crit(self, stat, model='c'):
        """
        Linear interpolation for Leybourne p-values and critical values

        Parameters
        ----------
        stat : float
            The Leybourne-McCabe test statistic
        model : {'c','ct'}
            The model used when computing the test statistic. 'c' is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated Leybourne-McCabe (KPSS) test statistic distribution
        """
        table = self.__leybourne_critical_values[model]
        # reverse the order
        y = table[:, 0]
        x = table[:, 1]
        # LM cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, x, y) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, np.flip(y), np.flip(x))
        cvdict = {"1%" : crit_value[0], "5%" : crit_value[1],
                    "10%" : crit_value[2]}
        return pvalue, cvdict

    def _tsls_arima(self, x, arlags, model):
        """
        Two-stage least squares approach for estimating ARIMA(p, 1, 1)
        parameters as an alternative to MLE estimation in the case of
        solver non-convergence

        Parameters
        ----------
        x : array_like
            data series
        arlags : int
            AR(p) order
        model : {'c','ct'}
            Constant and trend order to include in regression
            * 'c'  : constant only
            * 'ct' : constant and trend

        Returns
        -------
        arparams : int
            AR(1) coefficient plus constant
        theta : int
            MA(1) coefficient
        olsfit.resid : ndarray
            residuals from second-stage regression
        """
        endog = np.diff(x, axis=0)
        exog = lagmat(endog, arlags, trim='both')
        # add constant if requested
        if model == 'ct':
            exog = add_constant(exog)
        # remove extra terms from front of endog
        endog = endog[arlags:]
        if arlags > 0:
            resids = lagmat(OLS(endog, exog).fit().resid, 1, trim='forward')
        else:
            resids = lagmat(-endog, 1, trim='forward')
        # add negated residuals column to exog as MA(1) term
        exog = np.append(exog, -resids, axis=1)
        olsfit = OLS(endog, exog).fit()
        if model == 'ct':
            arparams = olsfit.params[1:(len(olsfit.params)-1)]
        else:
            arparams = olsfit.params[0:(len(olsfit.params)-1)]
        theta = olsfit.params[len(olsfit.params)-1]
        return arparams, theta, olsfit.resid

    def _autolag(self, x):
        """
        Empirical method for Leybourne-McCabe auto AR lag detection.
        Set number of AR lags equal to the first PACF falling within the
        95% confidence interval. Maximum nuber of AR lags is limited to
        the smaller of 10 or 1/2 series length.

        Parameters
        ----------
        x : array_like
            data series

        Returns
        -------
        arlags : int
            AR(p) order
        """
        p = pacf(x, nlags=min(int(len(x)/2), 10), method='ols')
        ci = 1.960 / np.sqrt(len(x))
        arlags = max(1, ([ n for n, i in enumerate(p) if abs(i) < ci ] + [-1])[0])
        return arlags

    def run(self, x, arlags=1, regression='c', method='mle', varest='var94'):
        """
        Leybourne-McCabe stationarity test

        The Leybourne-McCabe test can be used to test for stationarity in a
        univariate process.

        Parameters
        ----------
        x : array_like
            data series
        arlags : int
            number of autoregressive terms to include, default=None
        regression : {'c','ct'}
            Constant and trend order to include in regression
            * 'c'  : constant only (default)
            * 'ct' : constant and trend
        method : {'mle','ols'}
            Method used to estimate ARIMA(p, 1, 1) filter model
            * 'mle' : condition sum of squares maximum likelihood (default)
            * 'ols' : two-stage least squares
        varest : {'var94','var99'}
            Method used for residual variance estimation
            * 'var94' : method used in original Leybourne-McCabe paper (1994)
                        (default)
            * 'var99' : method used in follow-up paper (1999)

        Returns
        -------
        lmstat : float
            test statistic
        pvalue : float
            based on MC-derived critical values
        arlags : int
            AR(p) order used to create the filtered series
        cvdict : dict
            critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        H0 = series is stationary

        Basic process is to create a filtered series which removes the AR(p)
        effects from the series under test followed by an auxiliary regression
        similar to that of Kwiatkowski et al (1992). The AR(p) coefficients
        are obtained by estimating an ARIMA(p, 1, 1) model. Two methods are
        provided for ARIMA estimation: MLE and two-stage least squares.

        Two methods are provided for residual variance estimation used in the
        calculation of the test statistic. The first method ('var94') is the
        mean of the squared residuals from the filtered regression. The second
        method ('var99') is the MA(1) coefficient times the mean of the squared
        residuals from the ARIMA(p, 1, 1) filtering model.

        An empirical autolag procedure is provided. In this context, the number
        of lags is equal to the number of AR(p) terms used in the filtering
        step. The number of AR(p) terms is set equal to the to the first PACF
        falling within the 95% confidence interval. Maximum nuber of AR lags is
        limited to 1/2 series length.

        References
        ----------
        Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
        Testing the null hypothesis of stationarity against the alternative of
        a unit root. Journal of Econometrics, 54: 159–178.

        Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a
        unit root. Journal of Business and Economic Statistics, 12: 157–166.

        Leybourne, S.J., & McCabe, B.P.M. (1999). Modified stationarity tests
        with data-dependent model-selection rules. Journal of Business and
        Economic Statistics, 17: 264-270.

        Schwert, G W. (1987). Effects of model specification on tests for unit
        roots in macroeconomic data. Journal of Monetary Economics, 20: 73–103.
        """
        if regression not in ['c', 'ct']:
            raise ValueError(
                'LM: regression option \'%s\' not understood' % regression)
        if method not in ['mle', 'ols']:
            raise ValueError(
                'LM: method option \'%s\' not understood' % method)
        if varest not in ['var94', 'var99']:
            raise ValueError(
                'LM: varest option \'%s\' not understood' % varest)
        x = np.asarray(x)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
            raise ValueError(
                'LM: x must be a 1d array or a 2d array with a single column')
        x = np.reshape(x, (-1, 1))
        # determine AR order if not specified
        if arlags == None:
            arlags = self._autolag(x)
        elif not isinstance(arlags, int) or arlags < 1 or arlags > int(len(x) / 2):
            raise ValueError(
                'LM: arlags must be an integer in range [1..%s]' % str(int(len(x) / 2)))
        # estimate the reduced ARIMA(p, 1, 1) model
        if method == 'mle':
            arfit = ARIMA(x, order=(arlags, 1, 1), trend=regression).fit()
            resids = arfit.resid
            arcoeffs = arfit.arparams
            theta = arfit.maparams[0]
        else:
            arcoeffs, theta, resids = self._tsls_arima(x, arlags, model=regression)
        # variance estimator from (1999) LM paper
        var99 = abs(theta * np.sum(resids**2) / len(resids))
        # create the filtered series:
        #   z(t) = x(t) - arcoeffs[0]*x(t-1) - ... - arcoeffs[p-1]*x(t-p)
        z = np.full(len(x) - arlags, np.inf)
        for i in range(len(z)):
            z[i] = x[i + arlags]
            for j in range(len(arcoeffs)):
                z[i] -= arcoeffs[j] * x[i + arlags - j - 1]
        # regress the filtered series against a constant and
        # trend term (if requested)
        if regression == 'c':
            resids = z - z.mean()
        else:
            resids = OLS(z, add_constant(np.arange(1, len(z) + 1))).fit().resid
        # variance estimator from (1994) LM paper
        var94 = np.sum(resids**2) / len(resids)
        # compute test statistic with specified variance estimator
        eta = np.sum(resids.cumsum()**2) / (len(resids)**2)
        if varest == 'var99':
            lmstat = eta / var99
        else:
            lmstat = eta / var94
        # calculate pval
        crit = self.__leybourne_crit(lmstat, regression)
        lmpval = crit[0]
        cvdict = crit[1]
        return lmstat, lmpval, arlags, cvdict

    def __call__(self, x, arlags=None, regression='c', method='mle',
                 varest='var94'):
        return self.run(x, arlags=arlags, regression=regression, method=method,
                        varest=varest)

# output results
def _print_res(res, st):
    print("  lmstat =", "{0:0.5f}".format(res[0]), " pval =",
          "{0:0.5f}".format(res[1]), " arlags =", res[2])
    print("    cvdict =", res[3])
    print("    time =", "{0:0.5f}".format(time.time() - st))

# unit tests taken from Schwert (1987) and verified against Matlab
def main():
    print("Leybourne-McCabe stationarity test...")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "results\\")
    files = ['BAA.csv', 'DBAA.csv', 'SP500.csv', 'DSP500.csv', 'UN.csv', 'DUN.csv']
    lm = Leybourne()
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        st = time.time()
        if file == 'DBAA.csv':
            res = lm(mdl)
            _print_res(res=res, st=st)
            assert_equal(res[2], 3)
            assert_almost_equal(res[0], 0.1252, decimal=3)
            assert_almost_equal(res[1], 0.4747, decimal=3)
            st = time.time()
            res = lm(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 0.1248, decimal=3)
            assert_almost_equal(res[1], 0.0881, decimal=3)
            assert_equal(res[2], 3)
        elif file == 'DSP500.csv':
            res = lm(mdl)
            _print_res(res=res, st=st)
            assert_equal(res[2], 1)
            assert_almost_equal(res[0], 0.2855, decimal=3)
            assert_almost_equal(res[1], 0.1485, decimal=3)
            st = time.time()
            res = lm(mdl, varest='var99')
            _print_res(res=res, st=st)
            assert_equal(res[2], 1)
            assert_almost_equal(res[0], 0.2874, decimal=3)
            assert_almost_equal(res[1], 0.1468, decimal=3)
        elif file == 'DUN.csv':
            res = lm(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 0.1657, decimal=3)
            assert_almost_equal(res[1], 0.0348, decimal=3)
            st = time.time()
            res = lm(mdl, regression='ct', method='ols')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 0.1650, decimal=3)
            assert_almost_equal(res[1], 0.0353, decimal=3)
        elif file == 'BAA.csv':
            res = lm(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_equal(res[2], 4)
            assert_almost_equal(res[0], 2.4868, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
            st = time.time()
            res = lm(mdl, regression='ct', method='ols')
            _print_res(res=res, st=st)
            assert_equal(res[2], 4)
            assert_almost_equal(res[0], 2.9926, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
        elif file == 'SP500.csv':
            res = lm(mdl, arlags=4, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 1.8761, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
            st = time.time()
            res = lm(mdl, arlags=4, regression='ct', method='ols')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 1.9053, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
        elif file == 'UN.csv':
            res = lm(mdl, varest='var99')
            _print_res(res=res, st=st)
            assert_equal(res[2], 5)
            assert_almost_equal(res[0], 1221.0154, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
            st = time.time()
            res = lm(mdl, method='ols', varest='var99')
            _print_res(res=res, st=st)
            assert_equal(res[2], 5)
            assert_almost_equal(res[0], 1022.3827, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
