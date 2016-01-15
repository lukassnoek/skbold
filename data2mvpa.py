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

from __future__ import print_function, division
import os
import glob
import cPickle
import h5py
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from mvp_utils import sort_numbered_list
from os.path import join as opj
from nipype.interfaces import fsl


class Subject:

    def __init__(self, directory, mask_threshold=0, beta2tstat=True, ref_space='mni', mask_path=None, remove_class=[], cleanup=True):

        self.directory = directory
        self.sub_name = os.path.basename(os.path.dirname(directory))
        self.run_name = os.path.basename(directory).split('.')[0].split('_')[-1]
        self.ref_space = ref_space
        self.beta2tstat = beta2tstat
        self.mask_path = mask_path
        self.mask_threshold = mask_threshold
        self.cleanup = cleanup

        if mask_path is not None:
            self.mask_name = os.path.basename(os.path.dirname(mask_path))
        else:
            self.mask_name = 'WholeBrain'

        self.mask_index = None
        self.mask_shape = None

        self.class_labels = None
        self.n_class = None
        self.class_names = None
        self.remove_class = remove_class
        self.remove_idx = None

        self.n_trials = None
        self.n_features = None
        self.n_inst = None
        self.class_idx = None
        self.trial_idx = None

        self.nifti_header = None
        self.affine = None

        self.X = None
        self.y = None

    def extract_class_labels(self):
        """
        Extracts class-name of each trial and returns a vector of class labels.
        """

        sub_path = self.directory
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

    def convert2mni(self, file2transform):

        if type(file2transform) == str:
            base_path = os.path.dirname(file2transform)
            tmp = []
            tmp.append(file2transform)
            file2transform = tmp          
        elif type(file2transform) == list:
            base_path = os.path.dirname(file2transform[0])

        ref_file = opj(self.directory, 'reg', 'standard.nii.gz')
        matrix_file = opj(self.directory, 'reg', 'example_func2standard.mat')
        out_dir = opj(self.directory, 'reg_standard')

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out = []
        for f in file2transform:
            
            out_file = opj(out_dir, os.path.basename(f))
            apply_xfm = fsl.ApplyXfm()
            apply_xfm.inputs.in_file = f
            apply_xfm.inputs.reference = ref_file
            apply_xfm.inputs.in_matrix_file = matrix_file
            apply_xfm.interp = 'trilinear'
            apply_xfm.inputs.out_file = out_file
            apply_xfm.inputs.apply_xfm = True
            apply_xfm.run()
            out.append(out_file)

        return out

    def convert2epi(self, file2transform):

        if type(file2transform) == str:
            base_path = os.path.dirname(file2transform)
            tmp = []
            tmp.append(file2transform)
            file2transform = tmp
        elif type(file2transform) == list:
            base_path = os.path.dirname(file2transform[0])

        ref_file = opj(self.directory, 'mask.nii.gz')
        matrix_file = opj(self.directory, 'reg', 'standard2example_func.mat')
        out_dir = opj(self.directory, 'stats')

        out = []
        for f in file2transform:
            
            out_file = opj(out_dir, os.path.basename(f))
            apply_xfm = fsl.ApplyXfm()
            apply_xfm.inputs.in_file = f
            apply_xfm.inputs.reference = ref_file
            apply_xfm.inputs.in_matrix_file = matrix_file
            apply_xfm.interp = 'trilinear'
            apply_xfm.inputs.out_file = out_file
            apply_xfm.inputs.apply_xfm = True
            apply_xfm.run()

            out.append(out_file)

        return out

    def glm2mvp(self):

        sub_path = self.directory
        sub_name = self.sub_name
        run_name = self.run_name

        # Load mask, create index
        if self.mask_path is not None:

            mask_vol = nib.load(self.mask_path)
            
            if self.ref_space == 'epi' and mask_vol.shape == (91, 109, 91):
                self.mask_path = self.convert2epi(self.mask_path)[0]
                mask_vol = nib.load(self.mask_path)

            self.mask_shape = mask_vol.shape
            self.mask_index = mask_vol.get_data().ravel() > self.mask_threshold
            self.n_features = self.mask_index.sum()

        mat_dir = opj(os.path.dirname(sub_path), 'mvp_data')
        n_feat = len(glob.glob(opj(os.path.dirname(sub_path), '*.feat')))
        n_converted = len(glob.glob(opj(mat_dir, '*header*')))

        if os.path.exists(mat_dir) and self.cleanup and n_feat <= n_converted:
            shutil.rmtree(mat_dir)
            os.makedirs(mat_dir)
            n_converted = 0
        elif not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        # Extract class vector (class_labels)
        self.extract_class_labels()
        self.n_trials = len(self.class_labels)
        self.class_names = np.unique(self.class_labels)
        self.n_class = len(self.class_names)
        self.n_inst = [np.sum(cls == self.class_labels) for cls in self.class_names]
        self.class_idx = [self.class_labels == cls for cls in self.class_names]
        self.trial_idx = [np.where(self.class_labels == cls)[0] for cls in self.class_names]

        print('Processing %s (run %i / %i)...' % (sub_name, n_converted + 1, n_feat), end='')

        # Specify appropriate stats-directory
        if self.ref_space == 'epi':
            stat_dir = opj(sub_path, 'stats')
        elif self.ref_space == 'mni':
            stat_dir = opj(sub_path, 'reg_standard')
        else:
            raise ValueError('Specify valid reference-space (ref_space)')

        if self.ref_space == 'mni' and not os.path.isdir(stat_dir):
            stat_dir = opj(sub_path, 'stats')
            transform2mni = True
        else:
            transform2mni = False

        copes = glob.glob(opj(stat_dir, 'cope*.nii.gz'))
        copes = sort_numbered_list(copes)
        _ = [copes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        varcopes = glob.glob(opj(stat_dir, 'varcope*.nii.gz'))
        varcopes = sort_numbered_list(varcopes)
        _ = [varcopes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        n_stat = len(copes)

        if not n_stat == len(self.class_labels):
            msg = 'The number of trials (%i) do not match the number of class labels (%i)' % \
                  (n_stat, len(self.class_labels))
            raise ValueError(msg)

        # Transform to mni if necessary
        if transform2mni:
            copes.extend(varcopes)
            transformed_files = self.convert2mni(copes)            
            half = int(len(transformed_files) / 2)
            copes = transformed_files[:half]
            varcopes = transformed_files[half:]

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

        fn_header = opj(mat_dir, '%s_header_run%i.pickle' % (self.sub_name, n_converted+1))
        with open(fn_header, 'wb') as handle:
            cPickle.dump(self, handle)

        self.X = mvp_data
        fn_data = opj(mat_dir, '%s_data_run%i.hdf5' % (self.sub_name, n_converted+1))

        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=mvp_data)
        h5f.close()

        print(' done.')
        return self

    def merge_runs(self):

        mat_dir = opj(os.path.dirname(self.directory), 'mvp_data')

        run_headers = glob.glob(opj(mat_dir, '*pickle*'))
        run_data = glob.glob(opj(mat_dir, '*hdf5*'))

        if len(run_headers) > 1:
            print('Merging runs for %s' % self.sub_name)

            for i in range(len(run_data)):

                if i == 0:
                    h5f = h5py.File(run_data[i], 'r')
                    data = h5f['data'][:]
                    h5f.close()
                    hdr = cPickle.load(open(run_headers[i]))
                else:
                    tmp = h5py.File(run_data[i])
                    data = np.vstack((data, tmp['data'][:]))
                    tmp.close()
                    tmp = cPickle.load(open(run_headers[i], 'r'))
                    hdr.class_labels.extend(tmp.class_labels)

            fn_header = opj(mat_dir, '%s_header_merged.pickle' % self.sub_name)
            fn_data = opj(mat_dir, '%s_data_merged.hdf5' % self.sub_name)

            with open(fn_header, 'wb') as handle:
                cPickle.dump(hdr, handle)

            h5f = h5py.File(fn_data, 'w')
            h5f.create_dataset('data', data=data)
            h5f.close()
        else:
            pass


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

        mvp = cPickle.load(open(hdr_path[0]))
        h5f = h5py.File(data_path[0], 'r')
        mvp.X = h5f['data'][:]
        h5f.close()

        if self.shape == '4D':
            s = mvp.mask_shape

            # This is a really ugly hack, but for some reason the following
            # doesn't work: mvp.X.reshape((s[0],s[1],s[2],mvp.X.shape[0]))
            tmp_X = np.zeros((s[0], s[1], s[2], mvp.X.shape[0]))
            for trial in range(mvp.X.shape[0]):
                tmp_X[:, :, :, trial] = mvp.X[trial, :].reshape(mvp.mask_shape)
            mvp.X = tmp_X

        return mvp

    def write_4D_nifti(self):

        sub_name = os.path.basename(os.path.dirname(self.mvp_dir))

        print("Creating 4D nifti for %s" % sub_name)
        self.shape = '4D'
        mvp = self.load()
        img = nib.Nifti1Image(mvp.X, np.eye(4))
        nib.save(img, opj(self.mvp_dir, 'data_4d.nii.gz'))

        return self

class AverageSubject(Subject):
    """
    Will initialize a Subject object which contains the class-average patterns
    of a series of subjects, instead of a set of within-subject single-trial
    patterns.
    """

    def __init__(self):
        pass


class ConcatenatedSubject(object):
    """
    Will initialize a Subject object which contains a set of single-trial
    patterns concatenated across multiple subjects, yielding a matrix of
    (trials * subjects) x features.
    """
    
    def __init__(self, directory, identifier):

        self.directory = directory
        self.identifier = identifier
        self.name = 'ConcatenatedSubject'
        self.data_files = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.hdf5' % identifier))
        self.hdr_files = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.pickle' % identifier))

    def load(self):

        # Peek at first
        for i in range(len(self.data_files)):

            if i == 0:
                h5f = h5py.File(self.data_files[i], 'r')
                data = h5f['data'][:]
                h5f.close()
                hdr = cPickle.load(open(self.hdr_files[i]))

                if hdr.ref_space == 'epi':
                    raise ValueError('Cannot concatenate subjects from different (EPI) spaces!')

            else:
                tmp = h5py.File(self.data_files[i])
                data = np.vstack((data, tmp['data'][:]))
                tmp.close()
                tmp = cPickle.load(open(self.hdr_files[i], 'r'))
                hdr.class_labels.extend(tmp.class_labels)

        fn_header = opj(self.directory, '%s.pickle' % self.name)
        fn_data = opj(self.directory, '%s.hdf5' % self.name)

        with open(fn_header, 'wb') as handle:
            cPickle.dump(hdr, handle)

        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=data)
        h5f.close()        

        hdr.X = data

        return hdr