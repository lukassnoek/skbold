# Basic Mvp class, from which first-level specific (e.g. FSL or, perhaps in the
# future, SPM) containers/converters are subclassed.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, absolute_import, division

import os
import joblib
import nibabel as nib
import os.path as op
import numpy as np
import cPickle


class Mvp(object):
    """ Mvp (multiVoxel Pattern) class.

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).

    """

    def __init__(self, X=None, y=None, mask=None, mask_threshold=0):

        self.mask = mask
        self.mask_threshold = mask_threshold
        self.mask_shape = None
        self.nifti_header = None
        self.affine = None
        self.voxel_idx = None

        self.X = X
        self.y = y

    def write(self, path=None, name='mvp', backend='joblib'):

        if path is None:
            path = os.getcwd()

        fn = op.join(path, name)

        print("Saving file '%s' to disk." % fn)
        if backend == 'joblib':
            try:
                joblib.dump(self, fn + '.jl', compress=3)
            except:
                msg = "Array too large to save with joblib; using Numpy ... "
                print(msg)
                backend = 'numpy'

        if backend == 'numpy':
            np.save(fn + '_data.npy', self.X)

            with open(fn + '_header.pickle', 'wb') as hdr:
                self.X = None
                cPickle.dump(self, hdr)

    def load(self, path):
        # Load Mvp-object from disk
        pass

    def _update_mask_info(self, mask):

        mask_vol = nib.load(mask)
        mask_idx = mask_vol.get_data() > self.mask_threshold
        self.affine = mask_vol.affine
        self.nifti_header = mask_vol.header
        self.mask_shape = mask_vol.shape
        self.voxel_idx = np.arange(np.prod(self.mask_shape))
        self.voxel_idx = self.voxel_idx[mask_idx.ravel()]