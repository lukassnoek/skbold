# Class to reduce a whole-brain pattern to its ROI-average values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import glob
import nibabel as nib
import os.path as op
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from skbold.core import convert2mni, convert2epi

import skbold
roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs', 'harvard_oxford')

# To do: allow for functionality without mvp structure!


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
            mask_list = glob.glob(op.join(mask_dir, '*.nii.gz'))
        elif mask_type is 'bilateral':
            mask_dir = op.join(roi_dir, 'bilateral')
            mask_list = glob.glob(op.join(mask_dir, '*.nii.gz'))

        # If patterns are in epi-space, transform mni-masks to
        # subject specific epi-space if it doesn't exist already
        if mvp.ref_space == 'epi':
            epi_dir = op.join(op.dirname(mvp.directory), 'epi_masks', mask_type)
            reg_dir = op.join(mvp.directory, 'reg')
            print('Transforming mni-masks to epi (if necessary).')
            self.mask_list = convert2epi(mask_list, reg_dir, epi_dir)

        self.orig_mask = mvp.voxel_idx
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
            overlap = np.zeros(self.orig_shape).ravel()
            overlap[roi_idx.ravel()] += 1
            overlap[self.orig_mask] += 1
            region_av = np.mean(X[:, (overlap == 2)[self.orig_mask]], axis=1)
            X_new[:, i] = region_av

        return X_new