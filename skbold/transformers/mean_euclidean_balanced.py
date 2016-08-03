# Class to select features based on their mean euclidean distance between
# average class values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import warnings
from mean_euclidean import MeanEuclidean
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class MeanEuclideanBalanced(MeanEuclidean):
    """
    Implements transformation of features to average contrasts.
    This feature selection method calculates the average condition differences
    for all voxels, thresholds this, and averages the thresholded set to yield
    N(N-1)/2 features, in which N denotes the number of conditions.

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

    Notes
    -----
    The fit() method is documented in the MeanEuclidean documentation.
    """

    def __init__(self, cutoff=2.3, b_cutoff=100, normalize=False, fisher=False):

        super(MeanEuclideanBalanced, self).__init__(cutoff, normalize, fisher)
        self.b_cutoff = b_cutoff

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

        if self.b_cutoff < 10:
            self.idx_ = np.sum(self.condition_scores_ > self.b_cutoff, axis=0)
        else:
            sorted = np.argsort(self.condition_scores_, axis=1)
            self.idx_ = sorted[:, :self.b_cutoff].ravel()

        return X[:, self.idx_]