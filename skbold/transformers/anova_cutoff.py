# Class to implement sklearn's f_classif function, but with a minimum
# cutoff instead of an absolute or proportional amount of features.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class AnovaCutoff(BaseEstimator, TransformerMixin):
    """
    Implements ANOVA-based feature selection.
    Selects features based on an ANOVA F-test, but unlike existing
    implementations (e.g. sklearn's f_classif) this class implements a
    ANOVA-based feature selection based on a cutoff (minimal value) for the
    returned F-values.

    Parameters
    ----------
    cutoff : float or int
        Minimum F-value for feature to be included in the transform.
    """

    def __init__(self, cutoff=2.3):
        self.cutoff = cutoff
        self.scores_ = None
        self.idx_ = None

    def fit(self, X, y):
        """ Fits AnovaCutoff.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List[str] or numpy ndarray[str]
            List of ndarray with floats corresponding to labels
        """
        f, _ = f_classif(X, y)
        self.scores_ = f
        self.idx_ = f > self.cutoff

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
