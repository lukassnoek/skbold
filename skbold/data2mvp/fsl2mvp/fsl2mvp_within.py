# Class to extract and store first-level (meta)data from an FSL first-level
# (feat) directory.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
import cPickle
import h5py
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import os
import glob
import os.path as op
from skbold.utils import sort_numbered_list
from sklearn.preprocessing import LabelEncoder
from skbold.data2mvp.fsl2mvp import Fsl2mvp
from skbold.core import convert2epi, convert2mni


class Fsl2mvpWithin(Fsl2mvp):
    """ Fsl2mvp (multiVoxel Pattern) class, a subclass of Mvp (skbold.core)

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).
    """

    def __init__(self, directory, mask_threshold=0, beta2tstat=True,
                 ref_space='epi', mask_path=None, remove_cope=[], invert_selection=False):

        super(Fsl2mvpWithin, self).__init__(directory, mask_threshold, beta2tstat,
                                            ref_space, mask_path, remove_cope, invert_selection)


        self.n_trials = None
        self.n_features = None
        self.n_inst = None
        self.class_idx = None
        self.trial_idx = None


    def glm2mvp(self, extract_labels=True):
        """ Extract (meta)data from FSL first-level directory.

        This method extracts the class labels (y) and corresponding data
        (single-trial patterns; X) from a FSL first-level directory and
        subsequently stores it in the attributes of self.

        """
        sub_path = self.directory
        sub_name = self.sub_name

        reg_dir = op.join(sub_path, 'reg')

        # Load mask, create index
        if self.mask_path is not None:

            mask_vol = nib.load(self.mask_path)

            if self.ref_space == 'epi' and mask_vol.shape == (91, 109, 91):
                out_dir = reg_dir
                self.mask_path = convert2epi(self.mask_path, reg_dir, out_dir)[0]
                mask_vol = nib.load(self.mask_path)

            self.mask_shape = mask_vol.shape
            self.mask_index = mask_vol.get_data().ravel() > self.mask_threshold
            self.n_features = self.mask_index.sum()

        mat_dir = op.join(os.path.dirname(sub_path), 'mvp_data')
        n_feat = len(glob.glob(op.join(os.path.dirname(sub_path), '*.feat')))
        n_converted = len(glob.glob(op.join(mat_dir, '*header*')))

        if op.exists(mat_dir) and n_feat <= n_converted:
            shutil.rmtree(mat_dir)
            os.makedirs(mat_dir)
            n_converted = 0
        elif not op.exists(mat_dir):
            os.makedirs(mat_dir)

        # Extract class vector (cope_labels)
        if extract_labels:
            self._extract_labels()
            self.y = LabelEncoder().fit_transform(self.cope_labels)
            self.update_metadata()

        print('Processing %s (run %i / %i)...' % (sub_name, n_converted+1,
              n_feat), end='')

        # Specify appropriate stats-directory
        if self.ref_space == 'epi':
            stat_dir = op.join(sub_path, 'stats')
        elif self.ref_space == 'mni':
            stat_dir = op.join(sub_path, 'reg_standard')
        else:
            raise ValueError('Specify valid reference-space (ref_space)')

        if self.ref_space == 'mni' and not os.path.isdir(stat_dir):
            stat_dir = op.join(sub_path, 'stats')
            transform2mni = True
        else:
            transform2mni = False

        copes = glob.glob(op.join(stat_dir, 'cope*.nii.gz'))
        varcopes = glob.glob(op.join(stat_dir, 'varcope*.nii.gz'))
        copes, varcopes = sort_numbered_list(copes), sort_numbered_list(varcopes)

        # Transform (var)copes if ref_space is 'mni' but files are in 'epi'.

        if transform2mni:
            copes.extend(varcopes)
            out_dir = op.join(sub_path, 'reg_standard')
            transformed_files = convert2mni(copes, reg_dir, out_dir)
            half = int(len(transformed_files) / 2)
            copes = transformed_files[:half]
            varcopes = transformed_files[half:]

        _ = [copes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        varcopes = sort_numbered_list(varcopes)
        _ = [varcopes.pop(ix) for ix in sorted(self.remove_idx, reverse=True)]

        n_stat = len(copes)
        if not n_stat == len(self.cope_labels):
            msg = 'The number of trials (%i) do not match the number of ' \
                  'class labels (%i)' % (n_stat, len(self.cope_labels))
            raise ValueError(msg)

        # We need to 'peek' at the first cope to know the dimensions
        if self.mask_path is None:
            tmp = nib.load(copes[0]).get_data()
            self.n_features = tmp.size
            self.mask_index = np.ones(tmp.shape, dtype=bool).ravel()
            self.mask_shape = tmp.shape

        # Pre-allocate
        mvp_data = np.zeros((n_stat, self.n_features))

        # Load in data (COPEs)
        for i, path in enumerate(copes):
            cope_img = nib.load(path)
            mvp_data[i, :] = cope_img.get_data().ravel()[self.mask_index]

        self.nifti_header = cope_img.header
        self.affine = cope_img.affine

        if self.beta2tstat:
            for i_trial, varcope in enumerate(varcopes):
                var = nib.load(varcope).get_data()
                var_sq = np.sqrt(var.ravel()[self.mask_index])
                mvp_data[i_trial, :] = np.divide(mvp_data[i_trial, :], var_sq)

        mvp_data[np.isnan(mvp_data)] = 0

        fn_header = op.join(mat_dir, '%s_header_run%i.pickle' % (self.sub_name,
                            n_converted + 1))

        with open(fn_header, 'wb') as handle:
            cPickle.dump(self, handle)

        fn_data = op.join(mat_dir, '%s_data_run%i.hdf5' % (self.sub_name,
                          n_converted+1))

        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=mvp_data)
        h5f.close()

        print(' done.')

        return self

    def merge_runs(self, cleanup=True, iD='merged'):
        """ Merges single-trial patterns from different runs.

        Given m runs, this method merges patterns by simple concatenation.
        Concatenation either occurs along the vertical axis. Importantly,
        it assumes that runs are identical in their set-up (e.g., conditions).

        Parameters
        ----------
        cleanup : bool
            Whether to clean up the run-wise data and thus to keep only the
            merged data.
        id : str
            Identifier to give the merged runs, such that the data and header
            files have the structure of: <subname>_header/data_<id>.extension
        """

        mat_dir = op.join(op.dirname(self.directory), 'mvp_data')
        run_headers = glob.glob(op.join(mat_dir, '*pickle*'))
        run_data = glob.glob(op.join(mat_dir, '*hdf5*'))

        if len(run_headers) > 1:
            print('Merging runs for %s' % self.sub_name)

            for i in range(len(run_data)):

                # 'Peek' at first run
                if i == 0:
                    h5f = h5py.File(run_data[i], 'r')
                    data = h5f['data'][:]
                    h5f.close()
                    hdr = cPickle.load(open(run_headers[i]))
                else:
                    # Concatenate data to first run and extend cope_labels
                    tmp = h5py.File(run_data[i])
                    data = np.concatenate((data, tmp['data'][:]), axis=0)
                    tmp.close()
                    tmp = cPickle.load(open(run_headers[i], 'r'))
                    hdr.cope_labels.extend(tmp.cope_labels)

            hdr.update_metadata()
            hdr.y = LabelEncoder().fit_transform(hdr.cope_labels)

            fn_header = op.join(mat_dir, '%s_header_%s.pickle' %
                                (self.sub_name, iD))
            fn_data = op.join(mat_dir, '%s_data_%s.hdf5' %
                              (self.sub_name, iD))

            with open(fn_header, 'wb') as handle:
                cPickle.dump(hdr, handle)

            h5f = h5py.File(fn_data, 'w')
            h5f.create_dataset('data', data=data)
            h5f.close()

            if cleanup:
                run_headers.extend(run_data)
                _ = [os.remove(f) for f in run_headers]
        else:
            # If there's only one file, don't merge
            pass

    def glm2mvp_and_merge(self):
        """ Chains glm2mvp() and merge_runs(). """
        self.glm2mvp().merge_runs()
        return self