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
    Will implement a way to regress out a specified number of PCA components,
    which are assumed to be noise components.
    """

    def __init__(self, n_components=None, reject=None):

        self.n_components = n_components
        self.reject = reject
        self.pca = None

    def fit(self, X, y=None):

        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        self.pca = pca

        return self

    def transform(self, X):

        pca = self.pca
        X_pca = pca.transform(X)

        if self.reject is not None:
            to_reject = np.ones(X_pca.shape[1], dtype=bool)
            to_reject[self.reject] = False

            X_rec = X_pca[:, to_reject].dot(pca.components_[to_reject, :])
        else:
            X_rec = pca.inverse_transform(X_pca)

        return(X_rec)


if __name__ == '__main__':

    from skbold.utils import DataHandler
    from skbold import testdata_path, roidata_path
    from skbold.transformers import RoiIndexer
    from sklearn.decomposition import PCA
    mvp = DataHandler(identifier='merged').load_separate_sub(testdata_path)

    mask = op.join(roidata_path, 'harvard_oxford', 'bilateral', 'Amygdala.nii.gz')
    ri = RoiIndexer(mvp, mask)
    X = ri.fit_transform(mvp.X)

    pcafilt = PCAfilter(n_components=None, reject=[1, 2])
    X_filt = pcafilt.fit_transform(X)
