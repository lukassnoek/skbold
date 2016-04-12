# Class to index a whole-brain pattern with a certain ROI.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import glob
import os.path as op
import nibabel as nib
from sklearn.base import BaseEstimator, TransformerMixin
from ..core import convert2epi


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
        self.idx_ = None

        main_dir = op.dirname(mvp.directory)

        # Check if epi-mask already exists:
        if mvp.ref_space == 'epi':

            if mask[0:2] in ['L_', 'R_']:
                laterality = 'unilateral'
            else:
                laterality = 'bilateral'

            epi_dir = op.join(main_dir, 'epi_masks', laterality)
            if not op.isdir(epi_dir):
                os.makedirs(epi_dir)

            epi_name = op.basename(mask)
            epi_exists = glob.glob(op.join(epi_dir, '*%s*.nii.gz' % epi_name))
            if epi_exists:
                self.mask = epi_exists[0]
            else:
                reg_dir = op.join(mvp.directory, 'reg')
                self.mask = convert2epi(self.mask, reg_dir, epi_dir)[0]

    def fit(self, X=None, y=None):
        """ Fits RoiIndexer. """
        roi_idx = nib.load(self.mask).get_data() > self.mask_threshold
        overlap = roi_idx.astype(int).ravel() + self.orig_mask.astype(int)
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
