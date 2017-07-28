# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
from builtins import range
import os
import os.path as op
import skbold
import numpy as np
import nibabel as nib

from ..utils.roi_globals import available_atlases, other_rois
from ..utils.load_roi_mask import load_roi_mask, parse_roi_labels
from ..core import convert2epi
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from scipy.ndimage import label
from glob import glob
from warnings import warn

roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs',
                  'harvard_oxford')


class AverageRegionTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a whole-brain voxel pattern into a region-average pattern
    Computes the average from different regions from a given parcellation
    and returns those as features for X.

    Parameters
    ----------
    atlas : str
        Atlas to extract ROIs from. Available: 'HarvardOxford-Cortical',
        'HarvardOxford-Subcortical', 'HarvardOxford-All' (combination of
        cortical/subcortical), 'Talairach' (not tested), 'JHU-labels',
        'JHU-tracts', 'Yeo2011'.
    mvp : Mvp-object (see core.mvp)
        Mvp object that provides some metadata about previous masks
    mask_threshold : int (default: 0)
        Minimum threshold for probabilistic masks (such as Harvard-Oxford)
    reg_dir : str
        Path to directory with registration info (warps/transforms).
    **kwargs : key-word arguments
        Other arguments that can be passed to `skbold.utils.load_roi_mask`.
    """

    def __init__(self, atlas='HarvardOxford-All', mask_threshold=0, mvp=None,
                 reg_dir=None, orig_mask=None, data_shape=None, ref_space=None,
                 affine=None, **kwargs):

        if mvp is None:
            self.orig_mask = orig_mask
            self.data_shape = data_shape
            self.affine = affine
        else:
            self.orig_mask = mvp.voxel_idx
            self.data_shape = mvp.data_shape
            self.mask_threshold = mask_threshold
            self.affine = mvp.affine
            ref_space = mvp.ref_space

        rois, roi_names = load_roi_mask(roi_name='all', atlas_name=atlas,
                                        threshold=mask_threshold, **kwargs)

        self.roi_names = roi_names

        # This is actually very inefficient, because it warps all ROIs
        # separately, while it would be faster if just the atlas itself is
        # warped first
        if ref_space == 'epi':

            if reg_dir is None:
                warn('You have to provide a reg_dir because otherwise '
                     'we cannot transform masks to epi space.')

            to_transform = []
            for i, roi in enumerate(rois):
                img = nib.Nifti1Image(roi.astype(int), affine=self.affine)
                fn = op.join(reg_dir, 'roi_%i.nii.gz' % i)
                nib.save(img, fn)
                to_transform.append(fn)

            self.mask_list = convert2epi(to_transform, reg_dir, reg_dir)

    def fit(self, X=None, y=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """

        return self

    def transform(self, X, y=None):
        """ Transforms features from X (voxels) to region-average features.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : Optional[List[str] or numpy ndarray[str]]
            List of ndarray with strings indicating label-names

        Returns
        -------
        X_new : ndarray
            array with transformed data of shape = [n_samples, n_features]
            in which features are region-average values.
        """

        X_new = np.zeros((X.shape[0], len(self.mask_list)))
        for i, mask in enumerate(self.mask_list):

            roi_idx = nib.load(mask).get_data() > self.mask_threshold
            overlap = np.zeros(self.data_shape).ravel()
            overlap[roi_idx.ravel()] += 1
            overlap[self.orig_mask] += 1
            region_av = np.mean(X[:, (overlap == 2)[self.orig_mask]], axis=1)
            X_new[:, i] = region_av

        return X_new


class ClusterThreshold(BaseEstimator, TransformerMixin):
    """
    Implements a cluster-based feature selection method.
    This feature selection method performs a univariate feature selection
    method to yield a set of voxels which are then cluster-thresholded using
    a minimum (contiguous) cluster size. These clusters are then averaged to
    yield a set of cluster-average features. This method is described in detail
    in my master's thesis: github.com/lukassnoek/MSc_thesis.

    Parameters
    ----------
    transformer : scikit-learn style transformer class
        transformer class used to perform some kind of univariate feature
        selection.
    mvp : Mvp-object (see core.mvp)
        Necessary to provide mask metadata (index, shape).
    min_cluster_size : int
        minimum cluster size to be set for cluster-thresholding
    """

    def __init__(self, mvp, min_score, selector=f_classif,
                 min_cluster_size=20):

        self.min_score = min_score
        self.selector = selector
        self.min_cluster_size = min_cluster_size

        if hasattr(mvp, 'common_mask'):

            if mvp.common_mask is not None:
                mask_shape = mvp.common_mask['shape']
            else:
                mask_shape = mvp.data_shape
        else:
            mask_shape = mvp.data_shape

        self.mask_shape = mask_shape
        self.mask_idx = mvp.voxel_idx
        self.scores_ = None
        self.idx_ = None
        self.cl_idx_ = None
        self.n_clust_ = None

    def fit(self, X, y, *args):
        """ Fits ClusterThreshold transformer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List[str] or numpy ndarray[str]
            List of ndarray with floats corresponding to labels

        """

        self.scores_, _ = self.selector(X, y, *args)
        self.idx_ = self.scores_ > self.min_score

        # X_fs = univariate feature values in wholebrain space
        X_fs = np.zeros(self.mask_shape).ravel()
        X_fs[self.mask_idx] = self.scores_
        X_fs = X_fs.reshape(self.mask_shape)

        clustered, num_clust = label(X_fs > self.min_score)
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

        self.n_clust_ = cl_idx.shape[1]
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
        X_cl = np.zeros((X.shape[0], self.n_clust_))
        n_clust = X_cl.shape[1]

        for j in range(n_clust):
            idx = self.cl_idx_[:, j].astype(bool)
            X_cl[:, j] = np.mean(X[:, idx], axis=1)

        return X_cl


class PatternAverager(BaseEstimator, TransformerMixin):
    """
    Reduces the set of features to its average.

    Parameters
    ----------
    method : str
        method of averaging (either 'mean' or 'median')
    """

    def __init__(self, method='mean'):

        self.method = method

    def fit(self, X=None, y=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, X):
        """ Transforms patterns to its average.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X_new : ndarray
            Transformed ndarray of shape = [n_samples, 1]
        """

        if self.method == 'mean':
            X_new = np.mean(X, axis=1)
        elif self.method == 'median':
            X_new = np.median(X, axis=1)
        else:
            raise ValueError('Invalid method: choose mean or median.')

        return X_new


class PCAfilter(BaseEstimator, TransformerMixin):
    """
    Filters out a (set of) PCA component(s) and transforms it back to original
    representation.

    Parameters
    ----------
    n_components : int
        number of components to retain.
    reject : list
        Indices of components which should be additionally removed.

    Attributes
    ----------
    pca : scikit-learn PCA object
        Fitted PCA object.
    """

    def __init__(self, n_components=5, reject=None):

        self.n_components = n_components
        self.reject = reject
        self.pca = None

    def fit(self, X, y=None, *args):
        """ Fits PcaFilter.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List of str
            List or ndarray with floats corresponding to labels
        """

        pca = PCA(n_components=self.n_components, *args)
        pca.fit(X)
        self.pca = pca

        return self

    def transform(self, X):
        """ Transforms a pattern (X) by the inverse PCA transform with removed
        components.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X : ndarray
            Transformed array of shape = [n_samples, n_features] given the
            PCA calculated during fit().
        """

        pca = self.pca
        X_pca = pca.transform(X)

        if self.reject is not None:
            to_reject = np.ones(X_pca.shape[1], dtype=bool)
            to_reject[self.reject] = False

            X_rec = X_pca[:, to_reject].dot(pca.components_[to_reject, :])
        else:
            X_rec = pca.inverse_transform(X_pca)

        return X_rec

