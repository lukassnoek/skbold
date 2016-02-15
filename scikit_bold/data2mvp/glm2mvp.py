# -*- coding: utf-8 -*-
"""
Module with functions to load in first-level GLM estimates as 
returned by the software package FSL.

Lukas Snoek (lukassnoek@gmail.com)
"""

from __future__ import print_function, division, absolute_import
import os
import glob
import cPickle
import h5py
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scikit_bold.utils.mvp_utils import sort_numbered_list
from scikit_bold.transformers.transformers import *
from os.path import join as opj
from nipype.interfaces import fsl
from sklearn.preprocessing import LabelEncoder


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

        self.class_labels = ['_'.join(x.split('_')[:-1])
                             if x.split('_')[-1].isdigit() else x
                             for x in class_labels]

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
        # Remove annoying .mat files
        _ = [os.remove(f) for f in glob.glob(opj(os.getcwd(), '*.mat'))]

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
        self.y = LabelEncoder().fit_transform(self.class_labels)
        self.n_trials = len(self.class_labels)
        self.class_names = np.unique(self.class_labels)
        self.n_class = len(self.class_names)
        self.n_inst = [np.sum(cls == self.class_labels) for cls in self.class_names]
        self.class_idx = [self.class_labels == cls for cls in self.class_names]
        self.trial_idx = [np.where(self.class_labels == cls)[0] for cls in self.class_names]

        print('Processing %s (run %i / %i)...' % (sub_name, n_converted+1, n_feat), end='')

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
        varcopes = glob.glob(opj(stat_dir, 'varcope*.nii.gz'))

        if transform2mni:
            copes.extend(varcopes)
            transformed_files = self.convert2mni(copes)            
            half = int(len(transformed_files) / 2)
            copes = transformed_files[:half]
            varcopes = transformed_files[half:]

        copes = sort_numbered_list(copes)
        _ = [copes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        varcopes = sort_numbered_list(varcopes)
        _ = [varcopes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]

        n_stat = len(copes)
        if not n_stat == len(self.class_labels):
            msg = 'The number of trials (%i) do not match the number of class labels (%i)' % \
                  (n_stat, len(self.class_labels))
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

            hdr.y = LabelEncoder().fit_transform(hdr.class_labels)
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
    """ Loads in data/hdrs """
    def __init__(self, identifier='', shape='2D'):
        self.identifier = identifier
        self.shape = shape
        self.mvp = None

    def load_separate_sub(self, sub_dir):

        mvp_dir = opj(sub_dir, 'mvp_data')
        data_path = glob.glob(opj(mvp_dir, '*%s*.hdf5' % self.identifier))
        hdr_path = glob.glob(opj(mvp_dir, '*%s*.pickle' % self.identifier))

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

        self.mvp = mvp

        return mvp

    def load_concatenated_subs(self, directory):

        data_paths = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.hdf5' % self.identifier))
        hdr_paths = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.pickle' % self.identifier))

        # Peek at first
        for i in range(len(data_paths)):

            if i == 0:
                h5f = h5py.File(data_paths[i], 'r')
                data = h5f['data'][:]
                h5f.close()
                mvp = cPickle.load(open(hdr_paths[i]))

                if mvp.ref_space == 'epi':
                    raise ValueError('Cannot concatenate subjects from different (EPI) spaces!')

            else:
                tmp = h5py.File(data_paths[i])
                data = np.vstack((data, tmp['data'][:]))
                tmp.close()
                tmp = cPickle.load(open(hdr_paths[i], 'r'))
                mvp.class_labels.extend(tmp.class_labels)

        mvp.X = data
        mvp.sub_name = 'ConcatenatedSubjects'
        self.mvp = mvp

        return mvp

    def load_averaged_subs(self, directory):

        data_paths = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.hdf5' % self.identifier))
        hdr_paths = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.pickle' % self.identifier))

        # Peek at first
        for i in range(len(data_paths)):
            if i == 0:
                h5f = h5py.File(data_paths[i], 'r')
                data_tmp = h5f['data'][:]
                h5f.close()
                mvp = cPickle.load(open(hdr_paths[i]))

                # Pre-allocation
                data = np.zeros((len(data_paths), data_tmp.shape[0], data_tmp.shape[1]))
                data[i, :, :] = data_tmp

                if mvp.ref_space == 'epi':
                    raise ValueError('Cannot concatenate subjects from different (EPI) spaces!')

            else:
                tmp = h5py.File(data_paths[i])
                data[i, :, :] = tmp['data'][:]
                tmp.close()
                
        mvp.X = data.mean(axis=0)
        mvp.sub_name = 'AveragedSubjects'
        self.mvp = mvp
        return mvp

    def load_averagedcontrast_subs(self, directory, grouping):
        """
        Averages trials within conditions per subject and
        concatenates these condition-average patterns across
        subjects into one mvp-matrix.
        """

        data_paths = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.hdf5' % self.identifier))
        hdr_paths = glob.glob(opj(directory, '*', 'mvp_data', '*%s*.pickle' % self.identifier))

        # Loop over subjects
        for i in range(len(data_paths)):

            # peek at first subject (to get some meta-info)
            if i == 0:
                h5f = h5py.File(data_paths[i], 'r')
                data_tmp = h5f['data'][:]
                h5f.close()
                mvp = cPickle.load(open(hdr_paths[i]))

                if mvp.ref_space == 'epi':
                    raise ValueError('Cannot concatenate subjects from different (EPI) spaces!')

                # Group labels so we know which conditions (within factorial design) to average
                labfac = LabelFactorizer(grouping)
                mvp.y = labfac.fit_transform(mvp.class_labels)
                mvp.class_labels = list(labfac.get_new_labels())
                mvp.class_names = np.unique(mvp.class_labels)
                mvp.n_class = len(mvp.class_names)
                mvp.class_idx = [np.array(mvp.class_labels) == cls for cls in mvp.class_names]
                mvp.class_labels = list(mvp.class_names)    

                data_averaged = np.zeros((mvp.n_class, data_tmp.shape[1]))
                for ii in range(mvp.n_class):
                    data_averaged[ii, :] = np.mean(data_tmp[mvp.class_idx[ii], :], axis=0)

                # Pre-allocation
                data = np.zeros((len(data_paths) * mvp.n_class, data_tmp.shape[1]))
                data[(i*mvp.n_class):((i+1)*mvp.n_class), :] = data_averaged

            # This is executed in the rest of the loop
            else:
                tmp = h5py.File(data_paths[i])
                data_tmp = tmp['data'][:]
                tmp.close()
                hdr = cPickle.load(open(hdr_paths[i]))
                labfac = LabelFactorizer(grouping)
                hdr.y = labfac.fit_transform(hdr.class_labels)
                hdr.class_labels = list(labfac.get_new_labels())
                hdr.class_names = np.unique(hdr.class_labels)
                hdr.n_class = len(hdr.class_names)
                hdr.class_idx = [np.array(hdr.class_labels) == cls for cls in hdr.class_names]
                hdr.class_labels = hdr.class_names

                mvp.class_labels.extend(hdr.class_names)
                for ii in range(hdr.n_class):
                    # recycle data_averaged from when i==0

                    data_averaged[ii, :] = np.mean(data_tmp[mvp.class_idx[ii], :], axis=0)

                data[(i*mvp.n_class):((i+1)*mvp.n_class), :] = data_averaged

        mvp.X = data
        mvp.sub_name = 'AveragedContrastSubjects'
        self.mvp = mvp
        return mvp


    def write_4D_nifti(self):

        print("Creating 4D nifti for %s" % self.mvp.sub_name)
        mvp = self.load()
        img = nib.Nifti1Image(mvp.X, np.eye(4))
        nib.save(img, opj(self.mvp_dir, 'data_4d.nii.gz'))

        return self

if __name__ == '__main__':

    data = DataHandler().load_averagedcontrast_subs('/media/lukas/data/glm_mni', ['pos', 'neg'])