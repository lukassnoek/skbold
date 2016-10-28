from __future__ import print_function, division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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

        fids = np.unique(self.mvp.featureset_id)
        pos_idx = np.where(self.featureset_idx == fids)[0]

        mvp = self.mvp

        col_idx = np.in1d(mvp.featureset_id, self.featureset_idx)
        mvp.X = mvp.X[:, col_idx]
        mvp.voxel_idx = mvp.voxel_idx[col_idx]
        mvp.featureset_id = mvp.featureset_id[col_idx]
        mvp.featureset_id = np.zeros_like(mvp.featureset_id)
        mvp.data_shape = [mvp.data_shape[pos_idx]]
        mvp.data_name = [mvp.data_name[pos_idx]]
        mvp.affine = [mvp.affine[pos_idx]]

        self.mvp = mvp
        return mvp
