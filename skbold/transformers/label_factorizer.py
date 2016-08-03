# Class to 'factorize' labels into subgroups/classes.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LabelFactorizer(BaseEstimator, TransformerMixin):
    """ Transforms labels according to a given factorial grouping.

    Factorizes/encodes labels based on part of the string label. For example,
    the label-vector ['A_1', 'A_2', 'B_1', 'B_2'] can be grouped
    based on letter (A/B) or number (1/2).

    Parameters
    ----------
    grouping : List of str
        List with identifiers for condition names as strings

    Attributes
    ----------
    new_labels_ : list
        List with new labels.
    """

    def __init__(self, grouping):

        self.grouping = grouping
        self.new_labels_ = None

    def fit(self, y=None, X=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, y, X=None):
        """ Transforms label-vector given a grouping.

        Parameters
        ----------
        y : List/ndarray of str
            List of ndarray with strings indicating label-names
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        y_new : ndarray
            array with transformed y-labels
        X_new : ndarray
            array with transformed data of shape = [n_samples, n_features]
            given new factorial grouping/design.

        """
        y_new = np.zeros(len(y))*-1
        self.new_labels_ = np.array(['parsing error!'] * len(y))

        all_idx = np.zeros(len(y))
        for i, g in enumerate(self.grouping):
            idx = np.array([g in label for label in y])
            y_new[idx] = i
            self.new_labels_[idx] = g
            all_idx += idx

        # Index new labels, y, and X with new factorial labels
        all_idx = all_idx.astype(bool)
        y_new = y_new[all_idx]
        self.new_labels_ = self.new_labels_[all_idx]

        if X is not None:
            X_new = X[all_idx, :]
            return y_new, X_new

        return y_new

    def get_new_labels(self):
        """ Returns new labels based on factorization. """
        return self.new_labels_

