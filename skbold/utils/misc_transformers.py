# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ArrayPermuter(BaseEstimator, TransformerMixin):
    """ Permutes (shuffles) rows of matrix. """

    def __init__(self):
        """ Initializes ArrayPermuter object. """
        pass

    def fit(self, X=None, y=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, X):
        """ Permutes rows of data input.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X_new : ndarray
            ndarray with permuted rows

        """
        return np.random.permutation(X)


class RowIndexer(object):
    """
    Selects a subset of rows from an Mvp object.

    Notes
    -----
    NOT a scikit-learn style transformer.

    Parameters
    ----------
    idx : ndarray
        Array with indices.
    mvp : mvp-object
        Mvp-object to drawn metadata from.
    """

    def __init__(self, mvp, train_idx):
        self.idx = train_idx
        self.mvp = mvp

    def transform(self):
        """

        Returns
        -------
        mvp : mvp-object
            Indexed mvp-object.
        X_not_selected : ndarray
            Data which has not been selected.
        y_not_selected : ndarray
            Labels which have not been selected.
        """
        mvp = self.mvp
        selection = np.zeros(mvp.X.shape[0], dtype=bool)
        selection[self.idx] = True
        X_not_selected = mvp.X[~selection, :]
        y_not_selected = mvp.y[~selection]
        mvp.X = mvp.X[selection, :]
        mvp.y = mvp.y[selection]

        return mvp, X_not_selected, y_not_selected


class SelectFeatureset(BaseEstimator, TransformerMixin):
    """
    Selects only columns of a certain featureset.
    CANNOT be used in a scikit-learn pipeline!

    Parameters
    ----------
    mvp : mvp-object
        Used to extract meta-data.
    featureset_idx : ndarray
        Array with indices which map to unique feature-set voxels.
    """

    def __init__(self, mvp, featureset_idx):
        self.mvp = mvp
        self.featureset_idx = featureset_idx

    def fit(self):
        """ Does nothing, but included due to scikit-learn API. """
        return self

    def transform(self, X=None):
        """ Transforms mvp. """

        mvp = self.mvp
        fids = np.unique(mvp.featureset_id)
        col_idx = np.in1d(mvp.featureset_id, self.featureset_idx)
        pos_idx = np.where(col_idx)[0]

        if len(pos_idx) > 1:
            msg = ("Found more than one positional index when selecting "
                   "feature-set %i" % int(self.featureset_idx))
            raise ValueError(msg)
        elif len(pos_idx) == 0:
            msg = ("Didn't find a feature-set with id '%i'."
                   % self.featureset_idx)
            raise ValueError(msg)
        else:
            pos_idx = pos_idx[0]

        mvp.X = mvp.X[:, col_idx]
        mvp.voxel_idx = mvp.voxel_idx[col_idx]
        mvp.featureset_id = mvp.featureset_id[col_idx]
        mvp.featureset_id = np.zeros_like(mvp.featureset_id)
        mvp.data_shape = [mvp.data_shape[pos_idx]]
        mvp.data_name = [mvp.data_name[pos_idx]]
        mvp.affine = [mvp.affine[pos_idx]]

        self.mvp = mvp
        return mvp