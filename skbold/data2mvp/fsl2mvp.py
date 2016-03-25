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
from skbold.core import Mvp
from sklearn.preprocessing import LabelEncoder
from ..core import convert2mni, convert2epi


class Fsl2mvp(Mvp):
    """ Fsl2mvp (multiVoxel Pattern) class, a subclass of Mvp (scikit_bold.core)

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).

    To do: add feature to preserve order of presentation in trials!
    """

    def __init__(self, directory, mask_threshold=0, beta2tstat=True,
                 ref_space='epi', mask_path=None, remove_class=[]):

        super(Fsl2mvp, self).__init__(directory, mask_threshold, beta2tstat,
                                      ref_space, mask_path, remove_class)

    def extract_class_labels(self):
        """ Extracts class labels as strings from FSL first-level directory.

        This method reads in a design.con file, which is by default outputted
        in an FSL first-level directory, and sets self.class_labels to a list
        with labels, and in addition sets self.remove_idx with indices which
        trials (contrasts) were removed as indicated by the remove_class
        attribute from the Fsl2mvp object.

        """

        sub_path = self.directory
        sub_name = self.sub_name
        remove_class = self.remove_class

        design_file = op.join(sub_path, 'design.con')

        if not os.path.isfile(design_file):
            raise IOError('There is no design.con file for %s' % sub_name)

        # Find number of contrasts and read in accordingly
        contrasts = sum(1 if 'ContrastName' in line else 0
                        for line in open(design_file))

        n_lines = sum(1 for line in open(design_file))

        df = pd.read_csv(design_file, delimiter='\t', header=None,
                         skipfooter=n_lines-contrasts, engine='python')

        class_labels = list(df[1])

        # Remove to-be-ignored contrasts (e.g. cues)
        remove_idx = np.zeros((len(class_labels), len(remove_class)))

        for i, name in enumerate(remove_class):
            matches = [name in label for label in class_labels]
            remove_idx[:, i] = np.array(matches)

        self.remove_idx = np.where(remove_idx.sum(axis=1).astype(int))[0]
        _ = [class_labels.pop(idx) for idx in np.sort(self.remove_idx)[::-1]]

        # Here, numeric extensions of contrast names (e.g. 'positive_003') are
        # removed

        self.class_labels = []
        for c in class_labels:
            parts = c.split('_')
            parts = [x.strip() for x in parts]
            if parts[-1].isdigit():
                label = '_'.join(parts[:-1])
                self.class_labels.append(label)
            else:
                self.class_labels.append(c)

    def glm2mvp(self):
        """ Extract (meta)data from FSL first-level directory.

        This method extracts the class labels (y) and corresponding data
        (single-trial patterns; X) from a FSL first-level directory and
        subsequently stores it in the attributes of self.

        """
        sub_path = self.directory
        sub_name = self.sub_name
        run_name = self.run_name

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
        self.extract_class_labels()
        self.y = LabelEncoder().fit_transform(self.class_labels)
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
        if not n_stat == len(self.class_labels):
            msg = 'The number of trials (%i) do not match the number of ' \
                  'class labels (%i)' % (n_stat, len(self.class_labels))
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

        return self

    def glm2mvp_and_merge(self):
        """ Chains glm2mvp() and merge_runs(). """
        self.glm2mvp().merge_runs()
