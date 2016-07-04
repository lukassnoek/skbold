from __future__ import print_function, division
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, min_correlation=None, n_voxels=None):
        '''
        SEMI-TESTED!!!

        Parameters
        ----------
        min_correlation (int): if not None, all columns (voxels) with at least this value of pearson's r are selected
        n_voxels (int) : if not None, the maximum n_voxels correlations are selected.
        '''

        if (min_correlation==None and n_voxels==None) or (not min_correlation==None and not n_voxels==None):
            ValueError('Either choose minimal absolute correlation value, or top number of voxels; do not choose both.')

        self.min_correlation = min_correlation
        self.n_voxels = n_voxels


    def fit(self, X, y):

        correlations = np.apply_along_axis(pearsonr, 0, X, y)
        r_values = correlations[0,:]
        p_values = correlations[1,:]

        if not self.min_correlation==None:
            idx = np.abs(r_values) >= self.min_correlation
        if not self.n_voxels==None:
            r_values_abs = np.abs(r_values)
            idvals = np.argpartition(r_values_abs, -self.n_voxels)[-self.n_voxels:]
            idx = np.zeros(X.shape[1], dtype=bool)
            idx[idvals] = True

        self.idx_ = idx

        self.mvp.voxel_idx = self.mvp.voxel_idx[idx]
        return self

    def transform(self, X, y=None):

        Xnew = X[:, self.idx_]

        return Xnew