# Class to return the averaged pairwise contrast values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import warnings
import numpy as np
from .mean_euclidean import MeanEuclidean

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class FeaturesToContrast(MeanEuclidean):
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

    def __init__(self, cutoff=2.3, normalize=False, fisher=False):

        super(FeaturesToContrast, self).__init__(cutoff, normalize, fisher)

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

        X_new = np.zeros((X.shape[0], self.condition_idx_.shape[0]))
        for i in range(X_new.shape[1]):
            X_new[:, i] = np.mean(X[:, self.condition_idx_[i, :]], axis=1)

        return X_new
