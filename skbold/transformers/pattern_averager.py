# Class to return the averaged pattern (averaged features).

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PatternAverager(BaseEstimator, TransformerMixin):
    """
    Reduces the set of features to its average.

    Parameters
    ----------
    method : str
        method of averaging (either 'mean' or 'median')
    """

    def __init__(self, method='mean'):

        self.method = method

    def fit(self, X=None, y=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, X):
        """ Transforms patterns to its average.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X_new : ndarray
            Transformed ndarray of shape = [n_samples, 1]
        """

        if self.method == 'mean':
            X_new = np.mean(X, axis=1)
        elif self.method == 'median':
            X_new = np.median(X, axis=1)
        else:
            raise ValueError('Invalid method: choose mean or median.')

        return X_new