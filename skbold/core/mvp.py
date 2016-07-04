# Basic Mvp class, from which first-level specific (e.g. FSL or, perhaps in the
# future, SPM) containers/converters are subclassed.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, absolute_import, division

import nibabel as nib
import numpy as np
import os
import os.path as op
import cPickle
import h5py
import copy


class Mvp(object):
    """ Mvp (multiVoxel Pattern) class.

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).

    """

    def __init__(self, mask_threshold=0, run_name='generic', sub_name='generic',
                 ref_space='mni', mask_path=None, X=None, y=None):

        """ Initializes a (bare-bones) Mvp object.

        Parameters
        ----------
        directory : str
            Absolute path to directory from which first-level data should be
            extracted.
        mask_threshold : Optional[int or float]
            If a probabilistic mask is used, mask_threshold sets the lower-
            bound for the mask
        beta2tstat : bool
            Whether to convert extracted beta-values to t-statistics by
            dividing by their corresponding standard deviation.
        ref_space : str
            Indicates in which space the multivoxel patterns should be
            returned, either 'mni' (MNI152 2mm space) or 'epi' (native
            functional space). Thus far, MNI space only works for first-level
            data returned by fsl.
        mask_path : str
            Absolute path to the mask that will be used to index the patterns
            with.
        remove_class : list[str]
            List of condition names (or substrings of condition names) that
            need not to be included in the pattern-data (e.g. covariates,
            nuisance regressors, etc.).

        """

        self.run_name = run_name
        self.sub_name = sub_name
        self.ref_space = ref_space
        self.mask_path = mask_path
        self.mask_threshold = mask_threshold

        if mask_path is not None:
            self.mask_name = op.basename(mask_path).split('.')[0]
        else:
            self.mask_name = 'WholeBrain'

        self.n_features = None
        self.mask_index = None
        self.mask_shape = None
        self.nifti_header = None
        self.affine = None

        self.X = X
        self.y = y

        # Update some attributes if an mvp object is initialized
        # (and not a subclass!)
        if self.__class__.__name__ == 'Mvp':
            self._set_attributes()

    def _set_attributes(self):

        mask_vol = nib.load(self.mask_path)
        mask_data = mask_vol.get_data()
        self.mask_shape = mask_vol.shape
        self.nifti_header = mask_vol.header
        self.mask_index = (mask_data > self.mask_threshold).ravel()
        self.n_features = np.sum(self.mask_index)

    def update_mask(self, new_idx):

        if new_idx.size != self.mask_index.sum():
            msg = 'Shape of new index (%r) is not the same as the current ' \
                    'pattern (%r)!' % (new_idx.size, self.mask_index.sum())
            raise ValueError(msg)

        tmp_idx = np.zeros(self.mask_shape)
        tmp_idx[self.mask_index.reshape(self.mask_shape)] += new_idx
        self.mask_index = tmp_idx.astype(bool).ravel()

    def save(self, directory, name):

        fn_data = op.join(directory, '%s_data.hdf5' % name)
        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=self.X)
        h5f.close()

        self.X = None
        fn_header = op.join(directory, '%s_header.pickle' % name)
        with open(fn_header, 'wb') as handle:
            cPickle.dump(self, handle)

    def glm2mvp(self):
        msg = "This method can only be called by subclasses of Mvp!"
        raise ValueError(msg)

if __name__ == '__main__':

    import os.path as op
    base_dir = '/media/lukas/piop/GenderIntelligence'

    tbss_file_gender = op.join(base_dir, 'GenderDecoding', 'tbss_train_B.npy')
    tbss_file_intell = op.join(base_dir, 'IntelligenceDecoding', 'tbss_train_BG.npy')
    vbm_file_gender = op.join(base_dir, 'GenderDecoding', 'vbm_train_B.npy')
    vbm_file_intell = op.join(base_dir, 'IntelligenceDecoding', 'vbm_train_BG.npy')

    data = np.load(tbss_file_gender)
    mvp = Mvp(mask_path='/media/lukas/piop/GenderIntelligence/TBSS_mask.nii.gz',
              ref_space='mni1mm', mask_threshold=0, X=data)

    mni = np.zeros(mvp.mask_shape).ravel()
    mni[mvp.mask_index.ravel()] = data[0, :]
    mni = mni.reshape(mvp.mask_shape)
    img = nib.Nifti1Image(mni, affine=mvp.affine)
    img.to_filename('/home/lukas/test')