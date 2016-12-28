# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import os.path as op
import skbold
import numpy as np
import nibabel as nib

from ..core import convert2epi
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from scipy.ndimage import label
from glob import glob

roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs',
                  'harvard_oxford')


class ArrayPermuter(BaseEstimator, TransformerMixin):
    """ Permutes (shuffles) rows of matrix.
    """

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


class AverageRegionTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a whole-brain voxel pattern into a region-average pattern
    Computes the average from different regions from a given parcellation
    and returns those as features for X.

    Parameters
    ----------
    mask_type : List[str]
        List with absolute paths to nifti-images of brain masks in
        MNI152 (2mm) space.
    mvp : Mvp-object (see core.mvp)
        Mvp object that provides some metadata about previous masks
    mask_threshold : int (default: 0)
        Minimum threshold for probabilistic masks (such as Harvard-Oxford)
    """

    def __init__(self, mvp, mask_type='unilateral', mask_threshold=0):

        if mask_type is 'unilateral':
            mask_dir = op.join(roi_dir, 'unilateral')
            mask_list = glob(op.join(mask_dir, '*.nii.gz'))
        elif mask_type is 'bilateral':
            mask_dir = op.join(roi_dir, 'bilateral')
            mask_list = glob(op.join(mask_dir, '*.nii.gz'))

        # If patterns are in epi-space, transform mni-masks to
        # subject specific epi-space if it doesn't exist already
        if mvp.ref_space == 'epi':
            epi_dir = op.join(op.dirname(mvp.directory), 'epi_masks',
                              mask_type)
            reg_dir = op.join(mvp.directory, 'reg')
            print('Transforming mni-masks to epi (if necessary).')
            self.mask_list = convert2epi(mask_list, reg_dir, epi_dir)

        self.orig_mask = mvp.voxel_idx
        self.orig_shape = mvp.mask_shape
        self.orig_threshold = mvp.mask_threshold
        self.mask_threshold = mask_threshold

        _ = [os.remove(f) for f in
             glob(op.join(os.getcwd(), '*flirt.mat'))]

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
            overlap = np.zeros(self.orig_shape).ravel()
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
        self.mask_shape = mvp.mask_shape
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

        self.scores_ = self.selector(X, y, *args)
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


class RoiIndexer(BaseEstimator, TransformerMixin):
    """
    Indexes a whole-brain pattern with a certain ROI.
    Given a certain ROI-mask, this class allows transformation
    from a whole-brain pattern to the mask-subset.

    Parameters
    ----------
    mvp : mvp-object (see scikit_bold.core)
        Mvp-object, necessary to extract some pattern metadata
    mask : str
        Absolute paths to nifti-images of brain masks in MNI152 space.
    mask_threshold : Optional[int, float]
        Threshold to be applied on mask-indexing (given a probabilistic
        mask).
    """

    def __init__(self, mvp, mask, mask_threshold=0):

        self.mvp = mvp
        self.mask = mask
        self.mask_threshold = mask_threshold
        self.orig_mask = mvp.voxel_idx
        self.ref_space = mvp.ref_space
        self.idx_ = None

    def fit(self, X=None, y=None):
        """ Fits RoiIndexer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List of str
            List or ndarray with floats corresponding to labels
        """

        # Check if epi-mask already exists:
        if self.ref_space == 'epi':

            if op.basename(self.mask)[0:2] in ['L_', 'R_']:
                laterality = 'unilateral'
            else:
                laterality = 'bilateral'

            epi_dir = op.join(self.mvp.directory, 'epi_masks', laterality)

            if not op.isdir(epi_dir):
                os.makedirs(epi_dir)

            epi_name = op.basename(self.mask)[:-7]
            epi_exists = glob(op.join(epi_dir, '*%s*.nii.gz' % epi_name))

            if epi_exists:
                self.mask = epi_exists[0]
            else:
                reg_dir = op.join(self.mvp.directory, 'reg')
                self.mask = convert2epi(self.mask, reg_dir, epi_dir)[0]

        roi_idx = nib.load(self.mask).get_data() > self.mask_threshold
        overlap = np.zeros(self.mvp.mask_shape).ravel()
        overlap[roi_idx.ravel()] += 1
        overlap[self.orig_mask] += 1
        self.idx_ = (overlap == 2)[self.orig_mask]

        return self

    def transform(self, X, y=None):
        """ Transforms features from X (voxels) to a mask-subset.

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
        X_new = X[:, self.idx_]

        return X_new


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


class IncrementalFeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Indexes a set of features with a number of (sorted) features.
    Parameters
    ----------
    scores : ndarray
        Array of shape = n_features, or [n_features, n_class] in case of
        soft/hard voting in, e.g., a roi_stacking_classifier
        (see classifiers.roi_stacking_classifier).
    cutoff : int or float
        If int, it refers the absolute number of features included, sorted
        from high to low (w.r.t. scores). If float, it selects a proportion
        of features.
    """

    def __init__(self, scores, cutoff):

        self.scores = scores
        self.cutoff = cutoff
        self.idx_ = None

    def fit(self, X, y=None):
        """ Fits IncrementalFeatureCombiner transformer.
        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        """
        if self.cutoff >= 1:

            if self.scores.ndim > 1:
                mean_scores = self.scores.mean(axis=-1)
            else:
                mean_scores = self.scores

            best = np.argsort(mean_scores)[::-1][0:self.cutoff]
            self.idx_ = np.zeros(mean_scores.size, dtype=bool)
            self.idx_[best] = True

        else:
            self.idx_ = self.scores > self.cutoff

            if self.idx_.ndim > 1 and X.shape[1] == self.idx_.shape[0]:
                self.idx_ = self.idx_.sum(axis=1)

        if self.idx_.ndim > 1:
            self.idx_ = self.idx_.ravel()
        return self

    def transform(self, X, y=None):
        """ Transforms a pattern (X) given the indices calculated during fit().
        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        Returns
        -------
        X : ndarray
            Transformed array of shape = [n_samples, n_features] given the
            indices calculated during fit().
        """
        if self.idx_.size != X.shape[1]:
            n_class = X.shape[1] / self.idx_.size
            X_tmp = X.reshape((X.shape[0], n_class, self.idx_.size))
            X_tmp = X_tmp[:, :, self.idx_]
            return X_tmp.reshape((X.shape[0], np.prod(X_tmp.shape[1:])))
        else:
            return X[:, self.idx_]