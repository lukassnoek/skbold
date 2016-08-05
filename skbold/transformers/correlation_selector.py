# Class to select features based on a voxelwise correlation.
# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Performs univariate feature selection using voxelwise correlations.

    Parameters
    ----------
    min_correlation : int
        if not None, all columns (voxels) with at least this value of
        pearson's r are selected.
    n_voxels : int
        if not None, the maximum n_voxels correlations are selected.

    Attributes
    ----------
    idx : ndarray
        ndarray with indices of selected features.

    Raises
    ------
    ValueError
        If both min_correlation and n_voxels is selected.
    """

    def __init__(self, mvp, min_correlation=0.1, n_voxels=None,
                 by_featureset=False):

        no_choice = (min_correlation == None and n_voxels == None)
        both_choice = (not min_correlation == None and not n_voxels == None)
        if any([no_choice, both_choice]):
            msg = 'Either choose minimal absolute correlation value, ' \
                   'or top number of voxels; do not choose both.'
            ValueError(msg)

        self.min_correlation = min_correlation
        self.n_voxels = n_voxels
        self.mvp = mvp
        self.by_featureset = by_featureset

    def fit(self, X, y):
        """ Fits CorrelationSelector.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List[str] or numpy ndarray[str]
            List of ndarray with floats corresponding to labels
        """

        if self.n_voxels == 0:
            idx = np.ones(shape=self.mvp.X.shape[1], dtype=bool)
        else:
            if self.by_featureset:
                idx = np.empty(0)

                for f_set in np.unique(self.mvp.featureset_id):
                    correlations = np.apply_along_axis(pearsonr, 0, X, y)
                    r_values = correlations[0,:]
                    p_values = correlations[1,:]

                    if not self.min_correlation==None:
                        idx_this_fset = np.abs(r_values) >= self.min_correlation
                    if not self.n_voxels==None:
                        r_values_abs = np.abs(r_values)
                        idvals = np.argpartition(r_values_abs, -self.n_voxels)[-self.n_voxels:]
                        idx_this_fset = np.zeros(X.shape[1], dtype=bool)
                        idx_this_fset[idvals] = True
                idx = np.concatenate([idx, idx_this_fset])

            else:
                correlations = np.apply_along_axis(pearsonr, 0, X, y)
                r_values = correlations[0, :]
                p_values = correlations[1, :]

                if not self.min_correlation == None:
                    idx = np.abs(r_values) >= self.min_correlation
                if not self.n_voxels == None:
                    r_values_abs = np.abs(r_values)
                    idvals = np.argpartition(r_values_abs, -self.n_voxels)[-self.n_voxels:]
                    idx = np.zeros(X.shape[1], dtype=bool)
                    idx[idvals] = True

        self.idx_ = idx

        #Apply new indices to voxel_idx and contrast_id
        self.mvp.voxel_idx = self.mvp.voxel_idx[idx]

        if hasattr(self.mvp, 'featureset_id'):
            self.mvp.featureset_id = self.mvp.featureset_id[idx]

        return self

    def transform(self, X, y=None):
        """ Predicts X based on fitted CorrelationSelector.

         Parameters
         ----------
         X : ndarray
             Numeric (float) array of shape = [n_samples, n_features]
         y : List[str] or numpy ndarray[str]
             List of ndarray with floats corresponding to labels

         Returns
         -------
         X_new : ndarray
             array with transformed data of shape = [n_samples, n_features]
             in which features are voxels

         """

        X_new = X[:, self.idx_]

        return X_new