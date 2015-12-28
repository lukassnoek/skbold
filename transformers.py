"""
Module with transformer-classes following the scikit-learn API
Rewritten code from my MSc thesis project.
"""

from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'Lukas Snoek'


class Subject:
    """
    Contains (meta) data/patterns about single-trial fMRI-BOLD data.
    """

    def __init__(self, X, y, subject_name, mask_name, mask_index, mask_shape,
                 mask_threshold, class_labels):

        # Meta-data
        self.subject_name = subject_name
        self.class_labels = np.asarray(class_labels)
        self.class_names = np.unique(self.class_labels)

        # Primary data
        self.X = X
        self.y = y

        # Mask info
        self.mask_name = mask_name              # Name of nifti-file
        self.mask_index = mask_index            # index relative to MNI
        self.mask_shape = mask_shape            # shape of mask (usually mni)
        self.mask_threshold = mask_threshold

        # Information about condition/class
        self.n_class = len(np.unique(y))
        self.n_inst = [np.sum(cls == y) for cls in np.unique(y)]

        self.class_idx = [y == cls for cls in np.unique(y)]
        self.trial_idx = [np.where(y == cls)[0] for cls in np.unique(y)]


class AverageSubject:
    pass


"""
PRE-SPLIT TRANSFORMERS.
These classes transform the entire dataset, before the train/set split,
because they're not data-driven and thus do not need to be cross-validated.
"""


class AverageRegionTransformer():
    pass


class PCAfilter():
    pass


class Demean():
    pass


class SpatialFilter():
    pass


"""
POST-SPLIT TRANSFORMERS.
These classes estimate a transform based on the train-set only, which is then
applied (crossvalidated) on the test-set.
"""


class AnovaCutoff(BaseEstimator, TransformerMixin):
    pass


class ClusterThreshold(BaseEstimator, TransformerMixin):
    pass


class AveragePatterns(BaseEstimator, TransformerMixin):
    pass





class SelectAboveZvalue(BaseEstimator, TransformerMixin):
    """ Selects features based on normalized differentation scores above cutoff

    Feature selection method based on largest univariate differences;
    similar to sklearn's univariate feature selection (with fclassif), but
    selects all (normalized) difference scores, which are computed as the
    mean absolute difference between feature values averaged across classes*,
    above a certain z-value.
    * e.g. mean(abs(mean(A)-mean(B), mean(A)-mean(C), mean(B)-mean(C)))

    Works for any amount of classes.

    Attributes:
        zvalue (float/int): cutoff/lowerbound for normalized diff score

    """

    def __init__(self, zvalue):
        self.zvalue = zvalue
        self.idx_ = None
        self.zvalues_ = None

    def fit(self, X, y):
        """ Performs feature selection on array of n_samples, n_features
        Args:
            X: array with n_samples, n_features
            y: labels for n_samples
        """

        n_class = np.unique(y).shape[0]
        n_features = X.shape[1]

        av_patterns = np.zeros((n_class, n_features))

        # Calculate mean patterns
        for i in xrange(n_class):
            av_patterns[i, :] = np.mean(X[y == np.unique(y)[i], :], axis=0)

        # Create difference vectors, z-score standardization, absolute
        comb = list(itls.combinations(range(1, n_class + 1), 2))
        diff_patterns = np.zeros((len(comb), n_features))
        for i, cb in enumerate(comb):
            x = av_patterns[cb[0] - 1] - av_patterns[cb[1] - 1, :]
            diff_patterns[i, :] = np.abs((x - x.mean()) / x.std())

        mean_diff = np.mean(diff_patterns, axis=0)
        self.idx_ = mean_diff > self.zvalue
        self.zvalues_ = mean_diff

        return self

    def transform(self, X):
        """ Transforms n_samples, n_features array """
        return X[:, self.idx_]


