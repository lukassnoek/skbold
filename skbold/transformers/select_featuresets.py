from __future__ import print_function, division
import os
import glob
import os.path as op
import nibabel as nib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage.measurements import label


class SelectFeatureset(BaseEstimator, TransformerMixin):
    """ Selects only columns of a certain featureset. CANNOT be used in a pipeline!
    """

    def __init__(self, mvp, featureset_idx):
        self.mvp = mvp
        self.featureset_idx = featureset_idx

    def fit(self):
        ''' does nothing '''
        return self

    def transform(self, X=None):
        mvp = self.mvp

        col_idx = np.in1d(mvp.featureset_id, self.featureset_idx)
        mvp.X = mvp.X[:,col_idx]
        mvp.voxel_idx = mvp.voxel_idx[col_idx]
        mvp.featureset_id = mvp.featureset_id[col_idx]

        self.mvp = mvp
        return mvp

if __name__ == '__main__':
    import joblib
    import os.path as op

    mvp = joblib.load(op.join('/users/steven/Documents/Syncthing/MscProjects/Decoding/code/multimodal/MultimodalDecoding/data/between.jl'))

    selector = SelectFeatureset(mvp=mvp, featureset_idx = [1, 2])

    mvp = selector.fit().transform()

    print(mvp.featureset_id)
    print(mvp.voxel_idx)
    print(mvp.X)

    print(mvp.featureset_id.shape)
    print(mvp.voxel_idx.shape)
    print(mvp.X.shape)