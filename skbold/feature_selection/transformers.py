# This module contains transformers which are, strictly speaking,
# feature-selection transformers, and not feature-extraction transformers.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import os.path as op
from sklearn.base import BaseEstimator, TransformerMixin
from skbold.utils import load_roi_mask  # to prevent circular imports
from skbold.core import convert2epi
from glob import glob


class RoiIndexer(BaseEstimator, TransformerMixin):
    """
    Indexes a whole-brain pattern with a certain ROI.
    Given a certain ROI-mask, this class allows transformation
    from a whole-brain pattern to the mask-subset.

    Parameters
    ----------
    mvp : mvp-object (see scikit_bold.core)
        Mvp-object, necessary to extract some pattern metadata. If no mvp
        object has been supplied, you have to set which original mask has
        been used (e.g. graymatter mask) and what the reference-space is
        ('epi' or 'mni').
    mask : str
        Absolute paths to nifti-images of brain masks in MNI152 space
    mask_threshold : Optional[int, float]
        Threshold to be applied on mask-indexing (given a probabilistic
        mask).
    kwargs : key-word arguments
        Other arguments that will be passed to skbold's load_roi_mask function.
    """

    def __init__(self, mask, mask_threshold=0, mvp=None,
                 orig_mask=None, ref_space=None, reg_dir=None,
                 data_shape=None, affine=None, **kwargs):

        self.mvp = mvp
        self.mask = mask
        self.mask_threshold = mask_threshold
        self.reg_dir = reg_dir
        self.load_roi_args = kwargs

        if mvp is None:
            self.orig_mask = orig_mask
            self.ref_space = ref_space
            self.data_shape = data_shape
            self.affine = affine
        else:
            self.orig_mask = mvp.voxel_idx
            self.ref_space = mvp.ref_space
            self.data_shape = mvp.data_shape
            self.affine = mvp.affine

        if reg_dir is None and ref_space == 'epi':
            warn('Your data is in EPI space, but your mask is probably'
                 ' in MNI space, and you have not set the argument reg_dir. '
                 ' this is probably going to cause an error.')

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

        # If it's not an existing file, it's meant as a query for the internal
        # atlases
        if not op.isfile(self.mask):
            
            # Remove spaces because otherwise fsl might crash
            basename = self.mask.replace(' ', '_').replace(',', '')
            basename = basename.replace('(', '_').replace(')', '_')
            basename = basename.replace("'", '')

            mask, mask_name = load_roi_mask(basename,
                                            threshold=self.mask_threshold,
                                            **self.load_roi_args)
            if mask is None:
                raise ValueError("Could not find a mask for %s (given mask %s)" % 
                                 (basename, self.mask))

            self.mask = mask
            self.mask_name = mask_name

        # Check if epi-transformed mask already exists:
        if self.ref_space == 'epi':

            if not isinstance(self.mask, str):
                fn = op.join(self.reg_dir, self.mask_name + '.nii.gz')
                img = nib.Nifti1Image(self.mask.astype(int),
                                      affine=self.affine)
                nib.save(img, fn)
                self.mask = fn

            epi_name = op.basename(self.mask).split('.')[0]
            epi_exists = glob(op.join(self.reg_dir,
                                      '*%s_epi.nii.gz' % epi_name))
            if epi_exists:
                self.mask = epi_exists[0]
            else:

                self.mask = convert2epi(self.mask, self.reg_dir,
                                        self.reg_dir)

        roi_idx = nib.load(self.mask).get_data() > self.mask_threshold
        overlap = np.zeros(self.data_shape).ravel()
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
