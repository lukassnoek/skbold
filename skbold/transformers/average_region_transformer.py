# Class to reduce a whole-brain pattern to its ROI-average values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import glob
import nibabel as nib
import os.path as op
from ..data.ROIs import harvard_oxford as roi
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from nipype.interfaces import fsl
from ..core import convert2mni, convert2epi

# To do: allow for functionality without mvp structure!


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
            reg_dir = op.join(mvp.directory, 'reg')
            print('Transforming mni-masks to epi (if necessary).')
            self.mask_list = convert2epi(mask_list, reg_dir, epi_dir)

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