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
    """ Implements transformation of features to average contrasts.

    This feature selection method calculates the average condition differences
    for all voxels, thresholds this, and averages the thresholded set to yield
    N(N-1)/2 features, in which N denotes the number of conditions.

    """
    def __init__(self, cutoff=2.3, b_cutoff=100, normalize=False, fisher=False):
        """ Initializes FeaturesToContrast transformer.

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
        super(MeanEuclideanBalanced, self).__init__(cutoff, normalize, fisher)
        self.b_cutoff = b_cutoff

    def transform(self, X):

        if self.b_cutoff < 10:
            self.idx_ = np.sum(self.condition_scores_ > self.b_cutoff, axis=0)
        else:
            sorted = np.argsort(self.condition_scores_, axis=1)
            self.idx_ = sorted[:, :self.b_cutoff].ravel()

        return X[:, self.idx_]


if __name__ == '__main__':

    from skbold import mvp_test

    meb = MeanEuclideanBalanced(cutoff=2.3, b_cutoff=3)
    meb.fit(mvp_test.X, mvp_test.y)
    meb.transform(mvp_test.X)
    print(meb.idx_.sum())
