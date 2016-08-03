# Class to select features based on their mean euclidean distance between
# average class values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class MeanEuclidean(BaseEstimator, TransformerMixin):
    """
    Implements feature selection based on mean euclidian distance.
    This class implements a univariate feature selection method based on
    the largest condition-averaged euclidean distance.

    Parameters
    ----------
    cutoff : float or int
        Minimum average euclidean distance to be included in transformation
    normalize : bool
        Whether to normalize mean class activity by standard deviation
        across trials
    fisher : bool
        Whether to apply a fisher transform to the averaged euclidean
        distance.
    """

    def __init__(self, cutoff=2.3, normalize=False, fisher=False):
        self.cutoff = cutoff
        self.normalize = normalize
        self.fisher = fisher
        self.idx_ = None
        self.scores_ = None
        self.condition_idx_ = None
        self.condition_scores_ = None

    def fit(self, X, y):
        """ Fits MeanEuclidean transformer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List of str
            List or ndarray with floats corresponding to labels

        """

        n_class = np.unique(y).shape[0]
        n_features = X.shape[1]

        av_patterns = np.zeros((n_class, n_features))

        # Calculate mean patterns
        for i in range(n_class):
            pattern = X[y == np.unique(y)[i], :]

            if self.normalize:
                av_patterns[i, :] = pattern.mean(axis=0) / pattern.std(axis=0)
            else:
                av_patterns[i, :] = pattern.mean(axis=0)

        av_patterns[np.isnan(av_patterns)] = 0

        # Create difference vectors, z-score standardization, absolute
        comb = list(combinations(range(1, n_class + 1), 2))
        diff_patterns = np.zeros((len(comb), n_features))
        for i, cb in enumerate(comb):
            a, b = av_patterns[cb[0] - 1], av_patterns[cb[1] - 1, :]
            tmp = a - b

            if self.fisher:
                tmp = tmp / np.sqrt(a.std()**2 + b.std()**2)

            diff_patterns[i, :] = np.abs((tmp - tmp.mean()) / tmp.std())

        self.condition_idx_ = diff_patterns > self.cutoff
        self.condition_scores_ = diff_patterns
        mean_diff = np.mean(diff_patterns, axis=0)

        self.idx_ = mean_diff > self.cutoff
        self.scores_ = mean_diff

        return self

    def transform(self, X):
        """ Transforms a pattern (X) given the indices calculated during fit().

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

        return X[:, self.idx_]