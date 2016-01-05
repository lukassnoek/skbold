"""
Module with transformer-classes following the scikit-learn API
Rewritten code from my MSc thesis project (github.com/lukassnoek/MSc_thesis)
"""

from __future__ import division
import numpy as np
import nibabel as nib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from scipy.ndimage.measurements import label

__author__ = 'Lukas Snoek'


class Subject:
    """
    Contains meta data about single-trial fMRI-BOLD data.
    """

    def __init__(self, class_labels, subject_name, mask_name, mask_index,
                 mask_shape, mask_threshold):

        # Meta-data
        self.subject_name = subject_name
        self.class_labels = class_labels
        self.class_names = np.unique(self.class_labels)

        # Primary data
        self.X = None  # should be set in pipeline from hdf5 file
        self.y = None  # should be set in pipeline using LabelEncoder

        # Mask info
        self.mask_name = mask_name              # Name of nifti-file
        self.mask_index = mask_index            # index relative to MNI
        self.mask_shape = mask_shape            # shape of mask (usually mni)
        self.mask_threshold = mask_threshold

        # Information about condition/class
        self.n_class = len(self.class_names)
        self.n_inst = [np.sum(cls == class_labels) for cls in self.class_names]

        self.class_idx = [class_labels == cls for cls in self.class_names]
        self.trial_idx = [np.where(class_labels == cls)[0] for cls in self.class_names]


class AverageSubject(Subject):
    """
    Will initalize a Subject object which contains the class-average patterns
    of a series of subjects, instead of a set of within-subject single-trial
    patterns.
    """

    def __init__(self):
        pass

class ConcatenatedSubject(Subject):
    """
    Will initialize a Subject object which contains a set of single-trial
    patterns concatenated across multiple subjects, yielding a matrix of
    (trials * subjects) x features.
    """

    def __init__(self):
        pass


"""
PRE-SPLIT TRANSFORMERS.
These classes transform the entire dataset, before the train/set split,
because they're not data-driven and thus do not need to be cross-validated.
"""


class AverageRegionTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the average from different regions from a given parcellation
    and returns those as features for X.
    """
    def __init__(self, mask_list, orig_mask_index, orig_mask_threshold, orig_shape):
        self.mask_list = mask_list
        self.orig_mask = orig_mask_index
        self.orig_shape = orig_shape
        self.orig_threshold = orig_mask_threshold

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Transforms features from X (voxels) to region-average features from X.

        Args:
            X: array of trials x features (voxels in MNI152 2mm space)

        Returns:
            X_new: array of trials x region averages (of length mask_list)
        """

        X_new = np.zeros((X.shape[0], len(self.mask_list)))

        for i, mask in enumerate(self.mask_list):
            roi_idx = nib.load(mask).get_data() > self.orig_threshold
            overlap = roi_idx.astype(int).ravel() + self.orig_mask.astype(int)
            region_av = np.mean(X[:, (overlap == 2)[self.orig_mask]], axis=1)
            X_new[:, i] = region_av

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


class AnovaCutoff(BaseEstimator, TransformerMixin):
    """
    This class implements a ANOVA-based feature selection based on a
    cutoff (minimal value) for the returned F-values.
    """

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def fit(self, X, y):
        f, _ = f_classif(X, y)
        self.f_ = f
        self.idx_ = f > self.cutoff
        return self

    def transform(self, X):
        return X[:, self.idx_]


class ClusterThreshold(BaseEstimator, TransformerMixin):
    """
    Will implement a cluster-average feature selection as described in
    my master thesis (github.com/lukassnoek/MSc_thesis).
    """

    def __init__(self, cutoff, min_cluster_size, mask_shape, mask_idx):
        self.cutoff = cutoff
        self.min_cluster_size = min_cluster_size
        self.mask_shape = mask_shape
        self.mask_idx = mask_idx
        self.f_ = None
        self.idx_ = None
        self.cl_idx_ = None

    def fit(self, X, y):
        f, _ = f_classif(X, y)
        self.f_ = f
        self.idx_ = f > self.cutoff

        # X_fs = univariate feature values in wholebrain space
        X_fs = np.zeros(self.mask_shape).ravel()
        X_fs[self.mask_idx] = self.f_
        X_fs = X_fs.reshape(self.mask_idx)

        clustered, num_clust = label(X_fs > self.cutoff)
        values, counts = np.unique(clustered.ravel(), return_counts=True)
        n_clust = np.argmax(np.sort(counts)[::-1] < self.min_cluster_size)

        # Sort and trim
        cluster_nrs = values[counts.argsort()[::-1][:n_clust]]
        cluster_nrs = np.delete(cluster_nrs, 0)

        # cl_idx holds indices per cluster
        cl_idx = np.zeros((X_fs.size, len(cluster_nrs)))

        # Update cl_idx until cluster-size < cluster_min
        for j, clt in enumerate(cluster_nrs):
            cl_idx[:, j] = (clustered == clt).ravel()[self.mask_idx]

        self.cl_idx_ = cl_idx

        return self

    def transform(self, X):

        # X_cl = clustered version of X
        X_cl = np.zeros((X.shape[0], self.cl_idx_.shape[1]))
        n_clust = X_cl.shape[1]

        for j in xrange(n_clust):
            idx = self.cl_idx_[:, j]
            X_cl[:, j] = np.mean(X[:, idx], axis=1)

        return X_cl


class AveragePatterns(BaseEstimator, TransformerMixin):
    """
    Reduces the set of features to its average.
    """
    pass



