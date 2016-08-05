# Class to filter out a (set of) noisy PCA component(s).

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import os.path as op
import numpy as np


class PCAfilter(BaseEstimator, TransformerMixin):
    """
    Filters out a (set of) PCA component(s) and transforms it back to original
    representation.

    Parameters
    ----------
    n_components : int
        number of components to retain.
    reject : list
        Indices of components which should be additionally removed.

    Attributes
    ----------
    pca : scikit-learn PCA object
        Fitted PCA object.
    """

    def __init__(self, n_components=5, reject=None):

        self.n_components = n_components
        self.reject = reject
        self.pca = None

    def fit(self, X, y=None):
        """ Fits PcaFilter.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List of str
            List or ndarray with floats corresponding to labels
        """

        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        self.pca = pca

        return self

    def transform(self, X):
        """ Transforms a pattern (X) by the inverse PCA transform with removed
        components.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X : ndarray
            Transformed array of shape = [n_samples, n_features] given the
            PCA calculated during fit().
        """

        pca = self.pca
        X_pca = pca.transform(X)

        if self.reject is not None:
            to_reject = np.ones(X_pca.shape[1], dtype=bool)
            to_reject[self.reject] = False

            X_rec = X_pca[:, to_reject].dot(pca.components_[to_reject, :])
        else:
            X_rec = pca.inverse_transform(X_pca)

        return(X_rec)