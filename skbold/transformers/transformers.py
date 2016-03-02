# -*- coding: utf-8 -*-
""" Transformers module

This module contains transformer-classes following the scikit-learn API.
Contains (partly) rewritten code from my MSc thesis project
(github.com/lukassnoek/MSc_thesis).

Notes
-----
Depends on the scikit-learn BaseEstimator and TransformerMixin classes.

"""

from __future__ import print_function, division, absolute_import
import numpy as np
import nibabel as nib
import warnings
import glob
import os
import os.path as op
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from scipy.ndimage.measurements import label
from itertools import combinations
from joblib import Parallel, delayed
import skbold.ROIs.harvard_oxford as roi
from nipype.interfaces import fsl 

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

""" LABEL-TRANSFORMERS
These classes transform the dataset's labels/targets (y).
"""


class LabelFactorizer(BaseEstimator, TransformerMixin):
    """ Transforms labels according to a given factorial grouping.

    Factorizes/encodes labels based on part of the string label. For example,
    the label-vector ['A_1', 'A_2', 'B_1', 'B_2'] can be grouped
    based on letter (A/B) or number (1/2).
    """

    def __init__(self, grouping):
        """ Initializes LabelFactorizes with a given grouping.

        Parameters
        ----------
        grouping : List[str]
            List with identifiers for condition names as strings

        """
        self.grouping = grouping
        self.new_labels_ = None

    def fit(self, y=None, X=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, y, X=None):
        """ Transforms label-vector given a grouping.

        Parameters
        ----------
        y : List[str] or numpy ndarray[str]
            List of ndarray with strings indicating label-names
        X : Optional[ndarray]
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        y_new : ndarray
            array with transformed y-labels
        X_new : Optional[ndarray]
            array with transformed data of shape = [n_samples, n_features]
            given new factorial grouping/design.

        """
        y_new = np.zeros(len(y))*-1
        self.new_labels_ = np.array(['parsing error!'] * len(y))

        all_idx = np.zeros(len(y))
        for i, g in enumerate(self.grouping):
            idx = np.array([g in label for label in y])
            y_new[idx] = i
            self.new_labels_[idx] = g
            all_idx += idx

        # Index new labels, y, and X with new factorial labels
        all_idx = all_idx.astype(bool)
        y_new = y_new[all_idx]
        self.new_labels_ = self.new_labels_[all_idx]

        if X is not None:
            X_new = X[all_idx, :]
            return y_new, X_new

        return y_new

    def get_new_labels(self):
        """ Returns new labels based on factorization. """
        return self.new_labels_

""" PRE-SPLIT TRANSFORMERS.
These classes transform the entire dataset, before the train/set split,
because they're not data-driven and thus do not need to be cross-validated.
"""


class AverageRegionTransformer(BaseEstimator, TransformerMixin):
    """ Transforms a whole-brain voxel pattern into a region-average pattern

    Computes the average from different regions from a given parcellation
    and returns those as features for X.
    """

    def __init__(self, mvp, mask_type='unilateral', mask_threshold=0):
        """ Initializes AverageRegionTransformer object.

        Parameters
        ----------
        mask_type : List[str]
            List with absolute paths to nifti-images of brain masks in
            MNI152 (2mm) space.
        orig_mask : Optional[str]
            Path to the previous mask applied to the data (e.g. grey matter
            mask)
        orig_shape : Optional[tuple]
            Tuple with dimensions of original shape (before a mask was applied)
            assumed to be MNI152 (2mm) dimensions.
        orig_mask_threshold : Optional[int, float]
            Threshold used in previously applied mask (given a probabilistic
            mask)
        """

        if mask_type is 'unilateral':
            mask_dir = op.join(op.dirname(roi.__file__), 'unilateral')
            mask_list = glob.glob(op.join(mask_dir, '*.nii.gz'))
        elif mask_type is 'bilateral':
            mask_dir = op.join(op.dirname(roi.__file__), 'bilateral')
            mask_list = glob.glob(op.join(mask_dir, '*.nii.gz'))

        # If patterns are in epi-space, transform mni-masks to
        # subject specific epi-space if it doesn't exist already
        if mvp.ref_space == 'epi':
            epi_dir = op.join(op.dirname(mvp.directory), 'epi_masks', mask_type)
            if not op.isdir(epi_dir):
                os.makedirs(epi_dir)

            ref_file = op.join(mvp.directory, 'mask.nii.gz')
            matrix_file = op.join(mvp.directory, 'reg',
                                  'standard2example_func.mat')

            print('Transforming mni-masks to epi (if necessary).')
            for mask in mask_list:
                mask_file = op.basename(mask)
                epi_mask = op.join(epi_dir, mask_file)
                if not op.exists(epi_mask):

                    out_file = epi_mask
                    apply_xfm = fsl.ApplyXfm()
                    apply_xfm.inputs.in_file = mask
                    apply_xfm.inputs.reference = ref_file
                    apply_xfm.inputs.in_matrix_file = matrix_file
                    apply_xfm.interp = 'trilinear'
                    apply_xfm.inputs.out_file = out_file
                    apply_xfm.inputs.apply_xfm = True
                    apply_xfm.run()

            mask_list = glob.glob(op.join(epi_dir, '*.nii.gz'))
        self.mask_list = mask_list

        self.orig_mask = mvp.mask_index
        self.orig_shape = mvp.mask_shape
        self.orig_threshold = mvp.mask_threshold
        self.mask_threshold = mask_threshold

        _ = [os.remove(f) for f in glob.glob(op.join(os.getcwd(), '*flirt.mat'))]

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
            overlap = roi_idx.astype(int).ravel() + self.orig_mask.astype(int)
            region_av = np.mean(X[:, (overlap == 2)[self.orig_mask]], axis=1)
            X_new[:, i] = region_av

        return X_new


class RoiIndexer(BaseEstimator, TransformerMixin):
    """ Indexes a whole-brain pattern with a certain ROI.

    Given a certain ROI-mask, this class allows transformation
    from a whole-brain pattern to the mask-subset.
    """

    def __init__(self, mvp, mask, mask_threshold=0):
        """ Initializes RoiIndexer object.

        Parameters
        ----------
        mvp : mvp-object (see scikit_bold.core)
            Mvp-object, necessary to extract some pattern metadata
        mask : str
            Absolute paths to nifti-images of brain masks in MNI152 space.
        mask_threshold : Optional[int, float]
            Threshold to be applied on mask-indexing (given a probabilistic
            mask)
        """

        self.mask = mask
        self.mask_threshold = mask_threshold
        self.orig_mask = mvp.mask_index

        main_dir = op.dirname(mvp.directory)

        # Check if epi-mask already exists:
        if mvp.ref_space == 'epi':
            laterality = op.basename(op.dirname(mask))
            epi_dir = op.join(main_dir, 'epi_masks', laterality)
            if not op.isdir(epi_dir):
                os.makedirs(epi_dir)

            epi_name = op.basename(mask)
            epi_exists = glob.glob(op.join(epi_dir, epi_name))
            if epi_exists:
                self.mask = epi_exists[0]
            else:

                ref_file = op.join(mvp.directory, 'mask.nii.gz')
                matrix_file = op.join(mvp.directory, 'reg',
                                      'standard2example_func.mat')
                out_file = op.join(epi_dir, epi_name)
                apply_xfm = fsl.ApplyXfm()
                apply_xfm.inputs.in_file = self.mask
                apply_xfm.inputs.reference = ref_file
                apply_xfm.inputs.in_matrix_file = matrix_file
                apply_xfm.interp = 'trilinear'
                apply_xfm.inputs.out_file = out_file
                apply_xfm.inputs.apply_xfm = True
                apply_xfm.run()
                self.mask = out_file

    def fit(self, X=None, y=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
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

        roi_idx = nib.load(self.mask).get_data() > self.mask_threshold
        overlap = roi_idx.astype(int).ravel() + self.orig_mask.astype(int)
        X_new = X[:, (overlap == 2)[self.orig_mask]]
        return X_new


class PCAfilter(BaseEstimator, TransformerMixin):
    """
    Will implement a way to regress out a specified number of PCA components,
    which are assumed to be noise components.
    """
    pass


class SpatialFilter(BaseEstimator, TransformerMixin):
    """
    Will implement a spatial filter that high-passes a 3D pattern of
    voxel weights.
    """
    pass


"""
POST-SPLIT TRANSFORMERS.
These classes estimate a transform based on the train-set only, which is then
applied (crossvalidated) on the test-set.
"""


class ArrayPermuter(BaseEstimator, TransformerMixin):
    """ Permutes (shuffles) rows of matrix """

    def __init__(self):
        """ Initializes ArrayPermuter object. """
        self.shuffle = None

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


class AnovaCutoff(BaseEstimator, TransformerMixin):
    """ Implements ANOVA-based feature selection.

    Selects features based on an ANOVA F-test, but unlike existing
    implementations (e.g. sklearn's f_classif) this class implements a
    ANOVA-based feature selection based on a cutoff (minimal value) for the
    returned F-values.
    """

    def __init__(self, cutoff):
        """ Initializes AnovaCutoff transformer.

        Parameters
        ----------
        cutoff : float or int
            Minimum F-value for feature to be included in the transform.

        """
        self.cutoff = cutoff
        self.scores_ = None
        self.idx_ = None

    def fit(self, X, y):
        """ Fits AnovaCutoff.

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
        f, _ = f_classif(X, y)
        self.scores_ = f
        self.idx_ = f > self.cutoff

        return self

    def transform(self, X):
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
        return X[:, self.idx_]


class MeanEuclidean(BaseEstimator, TransformerMixin):
    """ Implements feature selection based on mean euclidian distance.

    This class implements a univariate feature selection method based on
    the largest condition-averaged euclidean distance.
    """

    def __init__(self, cutoff=2.3, normalize=False, fisher=False):
        """ Initializes MeanEuclidean transformer.

        Parameters
        ----------
        cutoff : float or int
            Minimum average euclidean distance to be included in transformation
        normalize : bool
            Whether to normalize mean class activity by standard deviation
            across trials
        fisher : bool
            Whether to apply a fisher transform to the averaged euclidean
            distance.

        """
        self.cutoff = cutoff
        self.normalize = normalize
        self.fisher = fisher
        self.idx_ = None
        self.condition_idx_ = None
        self.scores_ = None

    def fit(self, X, y):
        """ Fits MeanEuclidean transformer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List[str] or numpy ndarray[str]
            List of ndarray with floats corresponding to labels

        """

        n_class = np.unique(y).shape[0]
        n_features = X.shape[1]

        av_patterns = np.zeros((n_class, n_features))

        # Calculate mean patterns
        for i in range(n_class):
            pattern = X[y == np.unique(y)[i], :]

            if self.normalize:
                av_patterns[i, :] = pattern.mean(axis=0) / pattern.std(axis=0)
            else:
                av_patterns[i, :] = pattern.mean(axis=0)

        av_patterns[np.isnan(av_patterns)] = 0

        # Create difference vectors, z-score standardization, absolute
        comb = list(combinations(range(1, n_class + 1), 2))
        diff_patterns = np.zeros((len(comb), n_features))
        for i, cb in enumerate(comb):
            a, b = av_patterns[cb[0] - 1], av_patterns[cb[1] - 1, :]
            tmp = a - b

            if self.fisher:
                tmp = tmp / np.sqrt(a.std()**2 + b.std()**2)

            diff_patterns[i, :] = np.abs((tmp - tmp.mean()) / tmp.std())

        self.condition_idx_ = diff_patterns > self.cutoff
        mean_diff = np.mean(diff_patterns, axis=0)

        self.idx_ = mean_diff > self.cutoff
        self.scores_ = mean_diff

        return self

    def transform(self, X):
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
        return X[:, self.idx_]


class FeaturesToContrast(MeanEuclidean):
    """ Implements transformation of features to average contrasts.

    This feature selection method calculates the average condition differences
    for all voxels, thresholds this, and averages the thresholded set to yield
    N(N-1)/2 features, in which N denotes the number of conditions.

    """
    def __init__(self, cutoff=2.3, normalize=False, fisher=False):
        """ Initializes FeaturesToContrast transformer.

        Parameters
        ----------
        cutoff : float or int
            Minimum average euclidean distance to be included in transformation
        normalize : bool
            Whether to normalize mean class activity by standard deviation
            across trials
        fisher : bool
            Whether to apply a fisher transform to the averaged euclidean
            distance.
        """
        super(FeaturesToContrast, self).__init__(cutoff, normalize)

    def transform(self, X):
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

        X_new = np.zeros((X.shape[0], self.condition_idx_.shape[0]))
        for i in range(X_new.shape[1]):
            X_new[:, i] = np.mean(X[:, self.condition_idx_[i, :]], axis=1)

        return X_new


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


class AveragePatterns(BaseEstimator, TransformerMixin):
    """ Reduces the set of features to its average. """

    def __init__(self, method='mean'):
        """ Initializes AveragePatterns transformer.

        Parameters
        ----------
        method : str
            method of averaging (either 'mean' or 'median')

        """
        self.method = method

    def fit(self):
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


def fit_parallel(fold, X, y, pipeline, already_fitted=False):
    """ Should parallelize a fit-method """
    train_idx, test_idx = fold
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    if not already_fitted:
        pipeline.fit(X_train, y_train)

    return pipeline.predict_proba(X_test)


class VotingTransformer(BaseEstimator, TransformerMixin):
    """ Should implement a voting transformer. """
    def __init__(self, pipeline, folds, n_cores=1, already_fitted=False):

        self.pipeline = pipeline
        self.folds = folds
        self.n_cores = n_cores
        self.votes = None
        self.already_fitted = already_fitted

    def fit(self, X, y):

        probas = Parallel(n_jobs=self.n_cores)(delayed(fit_parallel)(fold, X, 
                          y, self.pipeline, self.already_fitted) for 
                          fold in self.folds)
        # probas = np.rollaxis(np.array(probas), 0, 3)
        # self.votes = np.argmax(np.mean(probas, axis=2), axis=1)
        # return self.votes

class LocalRegionCombiner():
    """ Should implement a class that combines local features in a meta-
    classifier. """
    pass
