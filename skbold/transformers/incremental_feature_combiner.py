# Class to select a (set of) feature(s) based on some external scoring metric.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import os.path as op
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class IncrementalFeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Indexes a set of features with a number of (sorted) features.

    Parameters
    ----------
    scores : ndarray
        Array of shape = n_features, or [n_features, n_class] in case of
        soft/hard voting in, e.g., a roi_stacking_classifier
        (see classifiers.roi_stacking_classifier).
    cutoff : int or float
        If int, it refers the absolute number of features included, sorted
        from high to low (w.r.t. scores). If float, it selects a proportion
        of features.
    """

    def __init__(self, scores, cutoff):

        self.scores = scores
        self.cutoff = cutoff
        self.idx_ = None

    def fit(self, X, y=None):
        """ Fits IncrementalFeatureCombiner transformer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        """
        if self.cutoff >= 1:

            if self.scores.ndim > 1:
                mean_scores = self.scores.mean(axis=-1)
            else:
                mean_scores = self.scores

            best = np.argsort(mean_scores)[::-1][0:self.cutoff]
            self.idx_ = np.zeros(mean_scores.size, dtype=bool)
            self.idx_[best] = True

        else:
            self.idx_ = self.scores > self.cutoff

            if self.idx_.ndim > 1 and X.shape[1] == self.idx_.shape[0]:
                self.idx_ = self.idx_.sum(axis=1)

        if self.idx_.ndim > 1:
            self.idx_ = self.idx_.ravel()
        return self

    def transform(self, X, y=None):
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
        if self.idx_.size != X.shape[1]:
            n_class = X.shape[1] / self.idx_.size
            X_tmp = X.reshape((X.shape[0], n_class, self.idx_.size))
            X_tmp = X_tmp[:, :, self.idx_]
            return X_tmp.reshape((X.shape[0], np.prod(X_tmp.shape[1:])))
        else:
            return X[:, self.idx_]
