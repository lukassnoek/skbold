"""
Module with transformer-classes following the scikit-learn API.
Contains rewritten code from my MSc thesis project
(github.com/lukassnoek/MSc_thesis).
"""

from __future__ import print_function, division, absolute_import
from scikit_bold.utils.mvp_utils import sort_numbered_list
import numpy as np
import nibabel as nib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from itertools import combinations
from os.path import join as opj
from nipype.interfaces import fsl
import os
import glob
import shutil
import pandas as pd
import cPickle
import h5py
from joblib import Parallel, delayed

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

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
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


class ArrayPermuter(BaseEstimator, TransformerMixin):
    """ Permutes (shuffles) rows of matrix """
    
    def __init__(self):
        self.shuffle = None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return np.random.permutation(X)

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


class MeanEuclidean(BaseEstimator, TransformerMixin):
    """
    Selects features based on mean Euclidean distance between mean patterns.
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
        for i in range(n_class):
            av_patterns[i, :] = np.mean(X[y == np.unique(y)[i], :], axis=0)

        # Create difference vectors, z-score standardization, absolute
        comb = list(combinations(range(1, n_class + 1), 2))
        diff_patterns = np.zeros((len(comb), n_features))
        for i, cb in enumerate(comb):
            tmp = av_patterns[cb[0] - 1] - av_patterns[cb[1] - 1, :]
            diff_patterns[i, :] = np.abs((tmp - tmp.mean()) / tmp.std())

        mean_diff = np.mean(diff_patterns, axis=0)
        self.idx_ = mean_diff > self.zvalue
        self.zvalues_ = mean_diff
        return self

    def transform(self, X):
        """ Transforms n_samples, n_features array """
        return X[:, self.idx_]


class FeaturesToContrast(BaseEstimator, TransformerMixin):
    
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
        for i in range(n_class):
            av_patterns[i, :] = np.mean(X[y == np.unique(y)[i], :], axis=0)

        # Create difference vectors, z-score standardization, absolute
        comb = list(combinations(range(1, n_class + 1), 2))
        diff_patterns = np.zeros((len(comb), n_features))
        for i, cb in enumerate(comb):
            tmp = av_patterns[cb[0] - 1] - av_patterns[cb[1] - 1, :]
            diff_patterns[i, :] = np.abs((tmp - tmp.mean()) / tmp.std())

        self.idx_ = diff_patterns > self.zvalue
        self.zvalues_ = diff_patterns
        return self

    def transform(self, X):

        X_new = np.zeros((X.shape[0], self.idx_.shape[0]))
        for i in range(X_new.shape[1]):
            X_new[:, i] = np.mean(X[:, self.idx_[i, :]], axis=1)
        return X_new

class ClusterThreshold(BaseEstimator, TransformerMixin):
    """
    Will implement a cluster-average feature selection as described in
    my master thesis (github.com/lukassnoek/MSc_thesis).
    """

    def __init__(self, mask_shape, mask_idx, cutoff=1, min_cluster_size=20):
        self.cutoff = cutoff
        self.min_cluster_size = min_cluster_size
        self.mask_shape = mask_shape
        self.mask_idx = mask_idx
        self.z_ = None
        self.idx_ = None
        self.cl_idx_ = None

    def fit(self, X, y):
        """
        something
        """
        transformer = MeanEuclidean(zvalue=self.cutoff)
        X_new = transformer.fit_transform(X, y)
        self.z_ = transformer.zvalues_
        self.idx_ = transformer.idx_

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

        # X_cl = clustered version of X
        X_cl = np.zeros((X.shape[0], self.cl_idx_.shape[1]))
        n_clust = X_cl.shape[1]

        for j in range(n_clust):
            idx = self.cl_idx_[:, j].astype(bool)
            X_cl[:, j] = np.mean(X[:, idx], axis=1)

        return X_cl


class AveragePatterns(BaseEstimator, TransformerMixin):
    """
    Reduces the set of features to its average.
    """

    def __init__(self, method='mean'):
        self.method = method

    def fit(self):
        return self

    def transform(self, X):

        if self.method == 'mean':
            X = np.mean(X, axis=1)
        elif self.method == 'median':
            X = np.median(X, axis=1)
        else:
            raise ValueError('Invalid method: choose mean or median.')

        return X


def fit_parallel(fold, X, y, pipeline, already_fitted=False):
    train_idx, test_idx = fold
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
            
    if not already_fitted:
        pipeline.fit(X_train, y_train)
    
    return pipeline.predict_proba(X_test)
    

class VotingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pipeline, folds, n_cores=1, already_fitted=False):

        self.pipeline = pipeline
        self.folds = folds
        self.n_cores = n_cores
        self.votes = None
        self.already_fitted = already_fitted
        
    def fit(self, X, y):

        probas = Parallel(n_jobs=self.n_cores)(delayed(fit_parallel)(fold, X, y, self.pipeline, self.already_fitted) for fold in self.folds)
        #probas = np.rollaxis(np.array(probas), 0, 3)
        #self.votes = np.argmax(np.mean(probas, axis=2), axis=1)
        #return self.votes

class LocalRegionCombiner():
    pass
