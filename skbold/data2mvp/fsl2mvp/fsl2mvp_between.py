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
from skbold.data2mvp.fsl2mvp import Fsl2mvp
from sklearn.preprocessing import LabelEncoder
from skbold.core import convert2mni, convert2epi


class Fsl2mvpBetween(Fsl2mvp):
    """ Fsl2mvp (multiVoxel Pattern) class, a subclass of Mvp (skbold.core)

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).
    """

    def __init__(self, directory, output_var_file=None, mask_threshold=0, beta2tstat=True,
                 ref_space='mni', mask_path=None, remove_cope=[], invert_selection=False):

        super(Fsl2mvpBetween, self).__init__(directory, mask_threshold, beta2tstat,
                                      ref_space, mask_path)

        self.output_var_file = output_var_file

        self.cope_labels = None
        self.n_cope = None
        self.cope_names = None
        self.remove_cope = remove_cope
        self.remove_idx = None
        self.invert_selection = invert_selection

        self.X_dict = {}
        self.X_labels = np.zeros(0, dtype=np.uint8)
        self.n_runs = None

    def _update_metadata(self):
        copes = self.cope_labels
        self.n_cope = len(copes)
        self.cope_names = np.unique(copes)

    def _update_X_dict(self, mvp_meta):
        for key, value in mvp_meta.iteritems():
            mvp_meta[key] = value + len(self.X_dict) * self.n_features

        self.X_dict.update(mvp_meta)

    def get_contrast(self, idx):
        '''
        Gets contrast names from list of idx

        Parameters
        ----------
        idx : int, indices of X

        Returns
        -------
        '''

        if type(idx) is not list:
            idx = [idx]

        labels = []
        for id in idx:
            label = [x[0] for x in self.X_dict.items() if id >= x[1][0] and id < x[1][1]]
            labels.append(label)

        return labels

    def _add_outcome_var(self, filename):
        file_path = op.join(op.dirname(self.directory), filename)

        with open(file_path, 'r') as f:
            y = float(f.readline())
        self.y = np.array(y)


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

        # Extract class vector (class_labels)
        self._extract_labels()
        # Update metadata, excluding X_dict
        self._update_metadata()

        print('Processing %s (run %i / %i)...' % (sub_name, n_converted + 1,
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

        if transform2mni:
            print('registering to mni...', end='')
            copes.extend(varcopes)
            out_dir = op.join(sub_path, 'reg_standard')
            transformed_files = convert2mni(copes, reg_dir, out_dir)
            half = int(len(transformed_files) / 2)
            copes = transformed_files[:half]
            varcopes = transformed_files[half:]

        _ = [copes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        varcopes = sort_numbered_list(varcopes)
        _ = [varcopes.pop(ix) for ix in sorted(self.remove_idx, reverse=True)]

        # if design_type=='within':
        #     n_stat = len(copes)
        #     if not n_stat == len(self.class_labels):
        #         msg = 'The number of trials (%i) do not match the number of ' \
        #               'class labels (%i)' % (n_stat, len(self.class_labels))
        #         raise ValueError(msg)
        # elif design_type=='between':
        #     n_stat = 1

        # We need to 'peek' at the first cope to know the dimensions
        if self.mask_path is None:
            tmp = nib.load(copes[0]).get_data()
            self.n_features = tmp.size
            self.mask_index = np.ones(tmp.shape, dtype=bool).ravel()
            self.mask_shape = tmp.shape

        columns = self.n_features * len(self.cope_labels)

        # Pre-allocate
        mvp_data = np.zeros((1, columns))
        mvp_meta = {} #empty dictionary for X_dict

        # Load in data (COPEs)
        for i, (cope, varcope) in enumerate(zip(copes, varcopes)):
            cope_img = nib.load(cope)
            copedat = cope_img.get_data().ravel()[self.mask_index]

            if self.beta2tstat:
                var = nib.load(varcope).get_data()
                var_sq = np.sqrt(var.ravel()[self.mask_index])
                copedat = np.divide(copedat, var_sq)

            mvp_data[0, (i * self.n_features):(i*self.n_features + self.n_features)] = copedat
            self.X_labels = np.concatenate((self.X_labels, np.ones(self.n_features, dtype=np.uint8) * i), axis=0)
            mvp_meta[self.cope_labels[i]] = np.array([(i * self.n_features), (i*self.n_features + self.n_features)])

        self.nifti_header = cope_img.header
        self.affine = cope_img.affine

        mvp_data[np.isnan(mvp_data)] = 0

        self._update_X_dict(mvp_meta)

        fn_header = op.join(mat_dir, '%s_header_run%i.pickle' % (self.sub_name,
                            n_converted+1))

        with open(fn_header, 'wb') as handle:
            cPickle.dump(self, handle)

        self.X = mvp_data

        fn_data = op.join(mat_dir, '%s_data_run%i.hdf5' % (self.sub_name,
                          n_converted+1))

        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=mvp_data)
        h5f.close()
        print(' done.')

    def merge_runs(self, cleanup=True, iD='merged'):
        """ Merges single-trial patterns from different runs.

        Given m runs, this method merges patterns by simple concatenation.
        Concatenation either occurs along the horizontal axis (if the design
        is between subjects) or along the vertical axis (if the design is within
        subjects). Importantly, for within subject designs, it assumes that
        runs are identical in their set-up (e.g., conditions).

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
                    data = np.concatenate((data, tmp['data'][:]), axis=1)
                    tmp.close()

                    tmp = cPickle.load(open(run_headers[i], 'r'))
                    hdr.cope_labels.extend(tmp.cope_labels)

                    hdr.X_labels = np.concatenate((hdr.X_labels, tmp.X_labels + len(np.unique(hdr.X_labels))), axis=0)

                    hdr._update_X_dict(tmp.X_dict)

            hdr._update_metadata()

            if self.output_var_file is not None:
                hdr._add_outcome_var(self.output_var_file)

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