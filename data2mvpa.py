# -*- coding: utf-8 -*-
"""
Module with functions to create subject-specific matrices of trials X voxels.

The creat_subject_mats function does the following for each subject:
1. Loads in mni-transformed first-level COPEs
2. Indexes vectorized copes with specified mask (e.g. ROI/gray matter)
3. Normalizes COPEs by their variance (sqrt(VARCOPE)); this will be extended 
with a multivariate normalization technique in the future
4. Initializes the result as an mvp_mat
5. Saves subject-specific mvp_mat as .cpickle file 

Lukas Snoek (lukassnoek@gmail.com)
"""

import os
import numpy as np
import nibabel as nib
import glob
import pickle
import h5py
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from mvp_utils import sort_numbered_list
#from nipype.interfaces import fsl
from transformers import Subject
from os.path import join as opj

__author__ = "Lukas Snoek"


def transform2mni(stat_paths, varcopes, sub_path):
    """
    Transforms (VAR)COPEs to MNI space.

    Args:
        stat_paths: list with paths to COPEs
        varcopes: list with paths to VARCOPEs
        sub_path: path to first-level directory

    Returns:
        stat_paths: transformed COPEs
        varcopes: transformed VARCOPEs
    """

    os.chdir(sub_path)
    print("Transforming COPES to MNI for %s." % sub_path)
    ref_file = opj(sub_path, 'reg', 'standard.nii.gz')
    field_file = opj(sub_path, 'reg', 'example_func2standard_warp.nii.gz')
    out_dir = opj(sub_path, 'reg_standard')

    for stat, varc in zip(stat_paths, varcopes):

        out_file = opj(out_dir, os.path.basename(stat))
        apply_warp = fsl.ApplyWarp()
        apply_warp.inputs.in_file = stat
        apply_warp.inputs.ref_file = ref_file
        apply_warp.inputs.field_file = field_file
        apply_warp.interp = 'trilinear'
        apply_warp.inputs.out_file = out_file
        apply_warp.run()

        out_file = opj(out_dir, os.path.basename(varc))
        apply_warp = fsl.ApplyWarp()
        apply_warp.inputs.in_file = varc
        apply_warp.inputs.ref_file = ref_file
        apply_warp.inputs.field_file = field_file
        apply_warp.interp = 'trilinear'
        apply_warp.inputs.out_file = out_file
        apply_warp.run()

    stat_dir = opj(sub_path, 'reg_standard')
    stat_paths = glob.glob(opj(stat_dir, 'cope*'))
    stat_paths = sort_numbered_list(stat_paths) # see function below

    varcopes = glob.glob(opj(stat_dir, 'varcope*'))
    varcopes = sort_numbered_list(varcopes) # see function below

    os.chdir('..')

    return stat_paths, varcopes


class DataHandler(object):

    def __init__(self, mvp_dir, identifier='', shape='2D'):
        self.mvp_dir = mvp_dir
        self.identifier = identifier
        self.shape = shape

    def load(self):

        data_path = glob.glob(opj(self.mvp_dir, '*%s*.hdf5' % self.identifier))
        hdr_path = glob.glob(opj(self.mvp_dir, '*%s*.pickle' % self.identifier))

        if len(data_path) > 1 or len(hdr_path) > 1:
            raise ValueError('Try to load more than one data/hdr file ...')

        mvp = pickle.load(open(hdr_path[0], 'rb'))
        h5f = h5py.File(data_path[0], 'r')
        mvp.X = h5f['data'][:]
        h5f.close()

        if self.shape == '4D':
            s = mvp.mask_shape

            # This is a really ugly hack, but for some reason the following
            # doesn't work: mvp.X.reshape((s[0],s[1],s[2],mvp.X.shape[0]))
            tmp_X = np.zeros((s[0], s[1], s[2], mvp.X.shape[0]))
            for trial in range(mvp.X.shape[0]):
                tmp_X[:,:,:,trial] = mvp.X[trial, :].reshape(mvp.mask_shape)
            mvp.X = tmp_X

        return mvp

    def write_4D_nifti(self):

        self.shape = '4D'
        mvp = self.load()
        img = nib.Nifti1Image(mvp.X, np.eye(4))
        nib.save(img, opj(self.mvp_dir, 'data_4d.nii.gz'))

        return self


class Fsl2mvp(object):

    def __init__(self, sub_path, mask=None, mask_threshold=0, remove_class=[],
                 ref_space='epi', beta2tstat=True, cleanup=1):

        self.sub_path = sub_path
        self.sub_name = os.path.basename(os.path.dirname(sub_path))
        self.run_name = os.path.basename(sub_path).split('.')[0].split('_')[-1]
        self.mask = mask
        self.mask_threshold = mask_threshold
        self.remove_class = remove_class
        self.ref_space = ref_space
        self.beta2tstat = beta2tstat
        self.cleanup = cleanup
        self.mvp = None # Should be Subject

    def extract_class_labels(self):
        """
        Extracts class-name of each trial and returns a vector of class labels.
        """

        sub_path = self.sub_path
        sub_name = self.sub_name
        remove_class = self.remove_class

        design_file = opj(sub_path, 'design.con')

        if not os.path.isfile(design_file):
            raise IOError('There is no design.con file for %s' % sub_name)

        # Find number of contrasts and read in accordingly
        contrasts = sum(1 if 'ContrastName' in line else 0 for line in open(design_file))
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
        self.class_labels = [s.split('_')[0] for s in class_labels]

        return self

    def transform(self):

        sub_path = self.sub_path
        sub_name = self.sub_name
        run_name = self.run_name

        # Load mask, create index
        if self.mask is not None:
            mask_name = os.path.basename(self.mask)
            mask_vol = nib.load(self.mask)
            mask_shape = mask_vol.shape
            mask_index = mask_vol.get_data().ravel() > mask_threshold
            n_features = mask_index.sum()

        mat_dir = opj(os.path.dirname(sub_path), 'mvp_data')
        n_feat = len(glob.glob(opj(os.path.dirname(sub_path), '*.feat')))
        n_converted = len(glob.glob(opj(mat_dir, '*header*')))
        run_name = os.path.basename(sub_path).split('.')[0]

        if os.path.exists(mat_dir) and self.cleanup and n_feat <= n_converted:
            shutil.rmtree(mat_dir)
            os.makedirs(mat_dir)
            n_converted = 0
        elif not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        # Extract class vector (class_labels)
        self.extract_class_labels()

        print('Processing %s (run %i / %i)...' % (sub_name, n_converted + 1, n_feat), end='')

        # Specify appropriate stats-directory
        if self.ref_space == 'epi':
            stat_dir = opj(sub_path, 'stats')
        elif self.ref_space == 'mni':
            stat_dir = opj(sub_path, 'reg_standard')
        else:
            raise ValueError('Specify valid reference-space (ref_space)')

        if self.ref_space == 'mni' and not os.path.isdir(stat_dir):
            # Here should be some mni-transformation code
            pass

        copes = glob.glob(opj(stat_dir, 'cope*.nii.gz'))
        copes = sort_numbered_list(copes)
        _ = [copes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        varcopes = glob.glob(opj(stat_dir, 'varcope*.nii.gz'))
        varcopes = sort_numbered_list(varcopes)
        _ = [varcopes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        n_stat = len(copes)

        if not n_stat == len(self.class_labels):
            msg = 'The number of trials do not match the number of class labels'
            raise ValueError(msg)

        if n_stat == 0:
            msg = 'There are no valid COPEs for subject %s.' % sub_name
            raise ValueError(msg)

        # We need to 'peek' at the first cope to know the dimensions
        if self.mask is None:
            tmp = nib.load(copes[0]).get_data()
            n_features = tmp.size
            mask_index = np.ones(tmp.shape, dtype=bool).ravel()
            mask_name = 'WholeBrain'
            mask_shape = tmp.shape

        # Pre-allocate
        mvp_data = np.zeros((n_stat, n_features))

        # Load in data (COPEs)
        for i, path in enumerate(copes):
            cope = nib.load(path).get_data().ravel()
            mvp_data[i, :] = cope[mask_index]

        if self.beta2tstat:
            for i_trial, varcope in enumerate(varcopes):
                var = nib.load(varcope).get_data()
                var_sq = np.sqrt(var.ravel()[mask_index])
                mvp_data[i_trial, :] = np.divide(mvp_data[i_trial, :], var_sq)

        mvp_data[np.isnan(mvp_data)] = 0

        # Initializing Subject object, which will be saved as a pickle file
        hdr = Subject(self.class_labels, sub_name, run_name, mask_name, mask_index,
                      mask_shape, self.mask_threshold)

        fn_header = opj(mat_dir, '%s_header_run%i.pickle' % (self.sub_name, n_converted+1))
        fn_data = opj(mat_dir, '%s_data_run%i.hdf5' % (self.sub_name, n_converted+1))

        with open(fn_header, 'wb') as handle:
            pickle.dump(hdr, handle)

        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=mvp_data)
        h5f.close()

        print(' done.')
        return self

    def merge_runs(self):

        mat_dir = opj(os.path.dirname(self.sub_path), 'mvp_data')

        run_headers = glob.glob(opj(mat_dir, '*pickle*'))
        run_data = glob.glob(opj(mat_dir, '*hdf5*'))

        if len(run_headers) > 1:
            print('Merging runs for %s' % self.sub_name)

            for i in range(len(run_data)):

                if i == 0:
                    h5f = h5py.File(run_data[i], 'r')
                    data = h5f['data'][:]
                    h5f.close()
                    hdr = pickle.load(open(run_headers[i], 'rb'))
                else:
                    tmp = h5py.File(run_data[i])
                    data = np.vstack((data, tmp['data'][:]))
                    tmp.close()
                    tmp = pickle.load(open(run_headers[i], 'rb'))
                    hdr.class_labels.extend(tmp.class_labels)

            fn_header = opj(mat_dir, '%s_header_merged.pickle' % self.sub_name)
            fn_data = opj(mat_dir, '%s_data_merged.hdf5' % self.sub_name)

            with open(fn_header, 'wb') as handle:
                pickle.dump(hdr, handle)

            h5f = h5py.File(fn_data, 'w')
            h5f.create_dataset('data', data=data)
            h5f.close()
        else:
            pass
