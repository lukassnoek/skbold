# Class to perform majority-class undersampling

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MajorityUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersamples the majority-class(es) by selecting random samples.

    Parameters
    ----------
    verbose : bool
        Whether to print downsamples number of samples.
    """

    def __init__(self, verbose=False):
        """ Initializes MajorityUndersampler object. """
        self.verbose = verbose
        self.idx_ = None

    def fit(self, X=None, y=None):
        """ Does nothing, but included for compatiblity with scikit-learn pipelines. """
        return self

    def transform(self, X, y):
        """ Downsamples majority-class(es).

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X : ndarray
            Transformed array of shape = [n_samples, n_features] given the
            indices calculated during fit().
        """

        bins = np.bincount(y)
        all_idx = np.zeros(y.size, dtype=bool)

        for i in np.unique(y):

            if bins[i] != np.min(bins):
                y_idx = y == i
                tmp_idx = np.zeros(y_idx.sum(), dtype=bool)
                tmp_idx[np.random.choice(np.arange(y_idx.sum()), np.min(bins), replace=False)] = True
                all_idx[y_idx] = tmp_idx
            else:
                all_idx[y == i] = True

        X_ds, y_ds = X[all_idx, :], y[all_idx]

        if self.verbose:
            print('Number of samples (after resampling): %.3f' % y_ds.size)
            print('Resampled class proportion: %.3f\n' % y_ds.mean())

        self.idx_ = all_idx

        return X[all_idx, :], y[all_idx]