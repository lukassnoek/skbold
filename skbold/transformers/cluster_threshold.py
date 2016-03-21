# Class to selected features based on a univariate feature selection which
# is subsequently cluster-thresholded
# (see https://github.com/lukassnoek/MSc_thesis).

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .mean_euclidean import MeanEuclidean
from scipy.ndimage.measurements import label

# To do: implement engine-option (fsl or scipy)


class ClusterThreshold(BaseEstimator, TransformerMixin):
    """ Implements a cluster-based feature selection method.

    This feature selection method performs a univariate feature selection
    method to yield a set of voxels which are then cluster-thresholded using
    a minimum (contiguous) cluster size. These clusters are then averaged to
    yield a set of cluster-average features. This method is described in detail
    in my master's thesis: github.com/lukassnoek/MSc_thesis.

    """

    def __init__(self, transformer=MeanEuclidean(cutoff=2.3, normalize=False,
                 fisher=False), mask_idx=None, mask_shape=(91, 109, 91),
                 min_cluster_size=20):
        """ Initializes ClusterThreshold transformer.

        Parameters
        ----------
        transformer : scikit-learn style transformer class
            transformer class used to perform univariate feature selection
        mask_idx : ndarray[bool]
            if a mask was applied before, this should give the indices used for
            the transformation
        mask_shape : tuple
            original size of the mask used for the initial transformation
            (assumed to be MNI152 2 mm dimensions)
        min_cluster_size : int
            minimum cluster size to be set for cluster-thresholding

        """
        self.transformer = transformer
        self.min_cluster_size = min_cluster_size
        self.mask_shape = mask_shape
        self.mask_idx = mask_idx
        self.z_ = None
        self.idx_ = None
        self.cl_idx_ = None

    def fit(self, X, y):
        """ Fits ClusterThreshold transformer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List[str] or numpy ndarray[str]
            List of ndarray with floats corresponding to labels

        """
        _ = self.transformer.fit_transform(X, y)
        self.z_ = self.transformer.zvalues_
        self.idx_ = self.transformer.idx_

        # X_fs = univariate feature values in wholebrain space
        X_fs = np.zeros(self.mask_shape).ravel()
        X_fs[self.mask_idx] = self.z_
        X_fs = X_fs.reshape(self.mask_shape)

        clustered, num_clust = label(X_fs > self.cutoff)
        values, counts = np.unique(clustered.ravel(), return_counts=True)
        n_clust = np.argmax(np.sort(counts)[::-1] < self.min_cluster_size)

        # Sort and trim
        cluster_nrs = values[counts.argsort()[::-1][:n_clust]]
        cluster_nrs = np.delete(cluster_nrs, 0)

        # cl_idx holds indices per cluster
        cl_idx = np.zeros((X.shape[1], len(cluster_nrs)))

        # Update cl_idx until cluster-size < cluster_min
        for j, clt in enumerate(cluster_nrs):
            cl_idx[:, j] = (clustered == clt).ravel()[self.mask_idx]

        self.cl_idx_ = cl_idx

        return self

    def transform(self, X):
        """ Transforms a pattern (X) given the indices calculated during fit().

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X_cl : ndarray
            Transformed array of shape = [n_samples, n_clusters] given the
            indices calculated during fit().

        """

        # X_cl = clustered version of X
        X_cl = np.zeros((X.shape[0], self.cl_idx_.shape[1]))
        n_clust = X_cl.shape[1]

        for j in range(n_clust):
            idx = self.cl_idx_[:, j].astype(bool)
            X_cl[:, j] = np.mean(X[:, idx], axis=1)

        return X_cl