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

from __future__ import division

import os
import numpy as np
import nibabel as nib
import glob
import cPickle

from mvp_utils import convert_labels2numeric, sort_numbered_list, \
    extract_class_vector, transform2mni

from transformers import Subject
from os.path import join as opj
from itertools import chain, izip

__author__ = "Lukas Snoek"


def create_subject_mats(sub_path, mask, mask_threshold, remove_class, grouping,
                        normalize_to_mni, beta2tstat):
    """ 
    Creates subject-specific mvp matrices, initializes them as an
    mvp_mat object and saves them as a cpickle file.
    
    Args: 
    firstlevel_dir  = directory with individual firstlevel data
    mask            = mask to index constrasts, either 'fstat' or a specific 
                      ROI. The exact name should be given 
                      (e.g. 'graymatter.nii.gz').
    subject_stem    = project-specific subject-prefix
    mask_threshold  = min. threshold of probabilistic FSL mask
    remove_class    = list with strings to match trials/classes to which need
                      to be removed (e.g. noise regressors)
   
    Returns:
    Nothing, but creates a dir ('mvp_mats') with individual pickle files.
    """

    # Load mask, create index
    mask_name = os.path.basename(mask)
    mask_vol = nib.load(mask)
    mask_shape = mask_vol.shape
    mask_index = mask_vol.get_data().ravel() > mask_threshold
    n_features = np.sum(mask_index)

    n_feat = len(glob.glob(opj(os.path.dirname(sub_path), '*.feat')))

    mat_dir = opj(os.path.dirname(sub_path), 'mvp_data')
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    # Extract class vector (see definition below) & convert to numeric
    class_labels, remove_idx = extract_class_vector(sub_path, remove_class)
    y = convert_labels2numeric(class_labels, grouping) # idea: use LabelEncoder!

    sub_name = os.path.basename(os.path.dirname(sub_path))
    print 'Processing %s ...' % sub_name

    # Generate and sort paths to stat files (COPEs/tstats)
    if normalize_to_mni:
        stat_dir = opj(sub_path, 'stats')
    else:
        stat_dir = opj(sub_path, 'reg_standard')

    copes = glob.glob(opj(stat_dir,'cope*.nii.gz'))
    copes = sort_numbered_list(copes)
    _ = [copes.pop(idx) for idx in sorted(remove_idx, reverse=True)]

    varcopes = glob.glob(opj(stat_dir,'varcope*.nii.gz'))
    varcopes = sort_numbered_list(varcopes)
    _ = [varcopes.pop(idx) for idx in sorted(remove_idx, reverse=True)]

    if normalize_to_mni:
        copes, varcopes = transform2mni(copes, varcopes, sub_path)

    n_stat = len(copes)

    if not n_stat == len(class_labels):
        msg = 'The number of trials do not match the number of class labels'
        raise ValueError(msg)

    if n_stat == 0:
        msg = 'There are no valid COPEs for subject %s.' % sub_name
        raise ValueError(msg)

    # Pre-allocate
    mvp_data = np.zeros([n_stat, n_features])

    # Load in data (COPEs)
    for i, path in enumerate(copes):
        cope = nib.load(path).get_data()
        mvp_data[i, :] = np.ravel(cope)[mask_index]

    if beta2tstat:
        for i_trial, varcope in enumerate(varcopes):
            var = nib.load(varcope).get_data()
            var_sq = np.sqrt(var.ravel()[mask_index])
            mvp_data[i_trial,] = np.divide(mvp_data[i_trial,], var_sq)

    mvp_data[np.isnan(mvp_data)] = 0

    # Initializing mvp_mat object, which will be saved as a pickle file
    to_save = Subject(mvp_data, y, sub_name, mask_name, mask_index,
                        mask_shape, mask_threshold, class_labels)

    fn_header = opj(mat_dir, '%s_header.cPickle' % sub_name)
    #fn_data = opj(mat_dir, '%s_data.hdf5' % sub_name)
    fn_data = opj(mat_dir, '%s_data.npy' % sub_name)

    with open(fn_header, 'wb') as handle:
        cPickle.dump(to_save, handle)

    np.save(fn_data, to_save)
    #h5f = h5py.File(fn_data, 'w')
    #h5f.create_dataset('data', data=mvp_data)
    #h5f.close()

    print 'Done processing %s.' % sub_name


def merge_runs():
    """
    Merges mvp_mat objects from multiple runs. 
    Incomplete; assumes only two runs for now.
    """

    header_paths = sorted(glob.glob(opj(os.getcwd(),'mvp_mats','*cPickle*')))
    header_paths = zip(header_paths[::2], header_paths[1::2])

    data_paths = sorted(glob.glob(opj(os.getcwd(),'mvp_mats','*hdf5*')))
    data_paths = zip(data_paths[::2], data_paths[1::2])

    sub_paths = zip(header_paths, data_paths)
    n_sub = len(sub_paths)

    for header, data in sub_paths:
        run1_h = cPickle.load(open(header[0]))
        run2_h = cPickle.load(open(header[1]))

        run1_d = h5py.File(data[0],'r')
        run2_d = h5py.File(data[1],'r')

        merged_grouping = run1_h.grouping
        merged_mask_index = run1_h.mask_index
        merged_mask_shape = run1_h.mask_shape
        merged_mask_name = run1_h.mask_name
        merged_mask_threshold = run1_h.mask_threshold
        merged_name = run1_h.subject_name.split('-')[0]

        merged_data = np.empty((run1_d['data'].shape[0] +
                                run2_d['data'].shape[0],
                                run1_d['data'].shape[1]))

        merged_data[::2,:] = run1_d['data'][:]
        merged_data[1::2,:] = run2_d['data'][:]

        merged_class_labels = list(chain.from_iterable(izip(run1_h.class_labels,
                                                            run2_h.class_labels)))

        merged_num_labels = list(chain.from_iterable(izip(run1_h.num_labels,
                                                          run2_h.num_labels)))

        to_save = Subject(merged_data, merged_name, merged_mask_name,
                            merged_mask_index, merged_mask_shape,
                            merged_mask_threshold, merged_class_labels,
                            np.asarray(merged_num_labels), merged_grouping)

        fn = opj(os.getcwd(), 'mvp_mats', merged_name + '_header_merged.cPickle')
        with open(fn, 'wb') as handle:
            cPickle.dump(to_save, handle)

        fn = opj(os.getcwd(), 'mvp_mats', merged_name + '_data_merged.hdf5')
        h5f = h5py.File(fn, 'w')
        h5f.create_dataset('data', data=merged_data)
        h5f.close()

        print "Merged subject %s " % merged_name

    #os.system('rm %s' % opj(os.getcwd(), 'mvp_mats', '*WIPPM*.cPickle'))
    #os.system('rm %s' % opj(os.getcwd(), 'mvp_mats', '*WIPPM*.hdf5'))
