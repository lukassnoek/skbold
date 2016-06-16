# Class to index a whole-brain pattern with a certain ROI.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import glob
import os.path as op
import nibabel as nib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..core import convert2epi


class MultiRoiIndexer(BaseEstimator, TransformerMixin):
    """ Wrapper that calls RoiIndexer multiple times for Fsl2mvpBetween mvps.
    """

    def __init__(self, mvp, maskdict, verbose=False):
        """ Initializes RoiIndexer object.

        Parameters
        ----------
        mvp : mvp-object (see scikit_bold.core)
            Mvp-object, necessary to extract some pattern metadata
        maskdict : dict of dicts
            dictionary with KEYS = COPE-names as they occur in self.X_dict;
            VALUE = dict with KEYS 'path' (absolute path to mask *.nii.gz file)
            and 'threshold' (threshold to be applied to that path)
        """

        self.mvp = mvp
        self.maskdict = maskdict
        self.orig_mask = mvp.mask_index
        self.directory = mvp.directory
        self.ref_space = mvp.ref_space
        self.idx_ = None
        self.verbose = verbose

    def fit(self, X=None, y=None):
        """ Fits multiple times. """

        cope_labels = self.mvp.cope_labels
        maskdict = self.maskdict

        #initialize roi_idx list
        roi_idx = np.ones(0, dtype=bool)

        for copeindex, cope in enumerate(cope_labels):
            if self.verbose:
                print('Cope: %s, path: %s, threshold: %f' %(cope, maskdict[cope]['path'], maskdict[cope]['threshold']))
            roi_idx_cope = nib.load(maskdict[cope]['path']).get_data() > maskdict[cope]['threshold']
            overlap = roi_idx_cope.astype(int).ravel() + self.orig_mask.astype(int)
            roi_idx_thiscope = (overlap==2)[self.orig_mask]
            roi_idx = np.hstack([roi_idx, roi_idx_thiscope])
            if self.verbose:
                print('Size of roi_idx: %f' %(roi_idx[roi_idx==True]).size)

        self.idx_ = roi_idx
        self.mvp.X_labels = self.mvp.X_labels[roi_idx]

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
