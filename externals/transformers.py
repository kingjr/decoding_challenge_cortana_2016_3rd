# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from nose.tools import assert_true

from sklearn.base import TransformerMixin, BaseEstimator

from mne.time_frequency import single_trial_power
from mne.filter import low_pass_filter, high_pass_filter, band_pass_filter


class _BaseEstimator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Baseliner(_BaseEstimator):
    def transform(self, X):
        if X.shape[-1] > 0:
            mean = np.mean(X, axis=-1)[..., None]
        else:
            mean = 0  # otherwise we get an ugly nan
        X -= mean
        return X


class TimeFreqDecomposer(_BaseEstimator):
    """Applies Wavelet Decomposition to signals and retrieve Power.

    See mne.time_frequency.single_trial_power
    """
    def __init__(self, sfreq, frequencies, n_cycles=5, decim=1, n_jobs=1):
        self.frequencies = np.array(frequencies)
        self.n_cycles = n_cycles
        self.decim = decim
        self.n_jobs = n_jobs
        self.sfreq = sfreq
        assert_true(isinstance(sfreq, (int, float)))
        assert_true((self.frequencies.ndim == 1) and len(self.frequencies))

    def transform(self, X):
        return single_trial_power(X, self.sfreq, self.frequencies,
                                  n_cycles=self.n_cycles, decim=self.decim,
                                  n_jobs=self.n_jobs)


class TimePadder(_BaseEstimator):
    """Padd time sample before and after each epoch to avoid edge artefacts.

    Parameters
    ----------
    n_sample : int
        Number of time samples to concatenate before and after each epoch.
    value : float
        Value of each padded time sample.
    """
    def __init__(self, n_sample, value=0.):
        self.n_sample = n_sample
        assert_true(isinstance(self.n_sample, int))
        self.value = value
        assert_true(isinstance(value, (int, float)))

    def transform(self, X):
        coefs = self.value * np.ones(X.shape[:2])
        coefs = np.tile(coefs, [self.n_sample, 1, 1]).transpose([1, 2, 0])
        X = np.concatenate((coefs, X, coefs), axis=2)
        return X


class TimeSelector(_BaseEstimator):
    """Select time sample in each epoch.

    Parameters
    ----------
    tslice : slice
        Slice of data from which to select. E.g. X[:, :, tslice]
    """
    def __init__(self, tslice):
        self.tslice = tslice
        assert_true(isinstance(self.tslice, (slice, int)))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = X[:, :, self.tslice]
        return X


class Reshaper(_BaseEstimator):
    """Transpose, concatenate and/or reshape data.

    Parameters
    ----------
    concatenate : int | None
        Reshaping feature dimension e.g. np.concatenate(X, axis=concatenate).
        Defaults to None.
    transpose : array of int, shape(1 + n_dims) | None
        Reshaping feature dimension e.g. X.transpose(transpose).
        Defaults to None.
    reshape : array, shape(n_dims) | None
        Reshaping feature dimension e.g. X.reshape(np.r_[len(X), shape]).
        Defaults to -1 if concatenate or transpose is None, else defaults
        to None.

    """

    def __init__(self, reshape=None, transpose=None, concatenate=None,
                 verbose=False):
        if (reshape is None) and (transpose is None) and (concatenate is None):
            reshape = [-1]
        self.reshape = reshape
        self.transpose = transpose
        self.concatenate = concatenate
        self.verbose = verbose

    def fit(self, X, y=None):
        self.shape_ = X.shape[1:]
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X, y=None):
        if self.transpose is not None:
            X = X.transpose(self.transpose)
        if self.concatenate:
            X = np.concatenate(X, self.concatenate)
        if self.reshape is not None:
            X = np.reshape(X, np.hstack((X.shape[0], self.reshape)))
        if self.verbose:
            print(self.shape_, '->', (X.shape[1:]))
        return X


class CustomEnsemble(TransformerMixin):
    """Stack multiple estimator predictions as if it were different features.

    Parameters
    ----------
    estimators : list of sklearn estimators.
        The estimators from which to predict.
    """
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y=None):
        for estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        all_Xt = list()
        for estimator in self.estimators:
            Xt = estimator.predict(X)
            all_Xt.append(Xt)
        all_Xt = np.c_[all_Xt].T
        return all_Xt

    def get_params(self, deep=True):
        return dict(estimators=self.estimators)


class Filterer(_BaseEstimator):
    """Filter signal
    See mne.filter.band_pass_filter
    """

    def __init__(self, sfreq, l_freq=None, h_freq=None, filter_length='10s',
                 l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
                 method='fft', iir_params=None):
        self.sfreq = sfreq
        self.l_freq = None if l_freq == 0 else l_freq
        self.h_freq = None if h_freq > (sfreq / 2.) else h_freq
        if (self.l_freq is not None) and (self.h_freq is not None):
            assert_true(self.l_freq < self.h_freq)
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.n_jobs = n_jobs
        self.method = method
        self.iir_params = iir_params
        assert_true((l_freq is None) or isinstance(l_freq, (int, float)))
        assert_true((h_freq is None) or isinstance(h_freq, (int, float)))

    def transform(self, X, y=None):

        kwargs = dict(Fs=self.sfreq, filter_length=self.filter_length,
                      method=self.method, iir_params=self.iir_params,
                      copy=False, verbose=False, n_jobs=self.n_jobs)
        if self.l_freq is None and self.h_freq is not None:
            filter_func = low_pass_filter
            kwargs['Fp'] = self.h_freq
            kwargs['trans_bandwidth'] = self.h_trans_bandwidth

        if self.l_freq is not None and self.h_freq is None:
            filter_func = high_pass_filter
            kwargs['Fp'] = self.l_freq
            kwargs['trans_bandwidth'] = self.l_trans_bandwidth

        if self.l_freq is not None and self.h_freq is not None:
            filter_func = band_pass_filter
            kwargs['Fp1'] = self.l_freq
            kwargs['Fp2'] = self.h_freq
            kwargs['l_trans_bandwidth'] = self.l_trans_bandwidth
            kwargs['h_trans_bandwidth'] = self.h_trans_bandwidth

        return filter_func(X, **kwargs)


class force_predict(_BaseEstimator):
    """Swap predict() method by another method.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator.
    method : str
        The swapping method, defaults to 'predict_proba'.
    axis : int, list, None
        The axis to give, default to 0.
    """
    def __init__(self, estimator, method='predict_proba', axis=0):
        self.method = method
        self.axis = axis
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        return self._force(X)

    def transform(self, X):
        return self._force(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _force(self, X):
        if self.method == 'predict_proba':
            proba = self.estimator.predict_proba(X)
            if self.axis == 'all':
                pass
            elif type(self.axis) in [int, list]:
                proba = proba[:, self.axis]
            return proba
        elif self.method == 'decision_function':
            distances = self.estimator.decision_function(X)
            if (len(distances.shape) == 1) or (self.axis == 'all'):
                pass
            elif type(self.axis) in [int, list]:
                distances = distances[:, self.axis]
            return distances
        else:
            return self.estimator.predict(X)
