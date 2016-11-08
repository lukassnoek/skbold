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
