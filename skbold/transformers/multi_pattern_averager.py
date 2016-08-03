# Class to return the averaged pattern (averaged features).

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MultiPatternAverager(BaseEstimator, TransformerMixin):
    """ Reduces the set of features to its average.

    Parameters
    ----------
    method : str
        method of averaging (either 'mean' or 'median')
    mvp : mvp-object
        Needed for metadata
    """

    def __init__(self, mvp, method='mean'):

        self.method = method
        self.mvp = mvp

    def fit(self, X=None, y=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, X, y=None):
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

        X_new = np.empty([X.shape[0], len(self.mvp.contrast_labels)])

        for copeindex, cope in enumerate(self.mvp.contrast_labels):
            if self.method == 'mean':
                current_cope_stat = X[:,self.mvp.contrast_id==copeindex].mean(1)
            elif self.method == 'median':
                current_cope_stat = X[:,self.mvp.contrast_id==copeindex].median(1)
            else:
                raise ValueError('Invalid method: choose mean or median.')
            X_new[:, copeindex] = current_cope_stat

        return X_new