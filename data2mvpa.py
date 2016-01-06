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
import pickle
import h5py
import pandas as pd
from mvp_utils import sort_numbered_list
#from nipype.interfaces import fsl
from transformers import Subject
from os.path import join as opj

__author__ = "Lukas Snoek"


def extract_class_vector(sub_path, ignore=[]):
    """
    Extracts class-name of each trial and returns a vector of class labels.

    Args:
        sub_path: path to subject-specific first-level directory
        ignore: name of contrasts that do not need to be read in (e.g. cues,
        nuisance regressors)

    Returns:
        class_labels: list of class-labels (strings)

    Raises:
        IOError: if design.con file doesn't exist
    """

    sub_name = os.path.basename(os.path.normpath(sub_path))
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
    remove_idx = np.zeros((len(class_labels), len(ignore)))

    for i, name in enumerate(ignore):
        matches = [name in label for label in class_labels]
        remove_idx[:, i] = np.array(matches)

    remove_idx = np.where(remove_idx.sum(axis=1).astype(int))[0]
    _ = [class_labels.pop(idx) for idx in np.sort(remove_idx)[::-1]]
    class_labels = [s.split('_')[0] for s in class_labels]

    return class_labels, remove_idx


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


def glm2mvpa(sub_path, mask, mask_threshold=0, remove_class=[],
             normalize_to_mni=False, beta2tstat=True):
    """
    Per subject, loads in first-level single-trial beta-estimates (FSL COPEs),
    optionally transforms these to MNI-space and converts them to t-statistics,
    and saves them as a trials x voxels array.

    Args:
        sub_path: path to first-level directory
        mask: mask to index COPEs with (e.g. all grey matter)
        mask_threshold: minimum threshold for probabilistic mask
        remove_class: list of strings corresponding to regressors to omit
        normalize_to_mni: whether to transform COPEs to MNI152 space
        beta2tstat: whether to convert betas to t-statistics (b/sqrt(cope)).
    """

    # Load mask, create index
    mask_name = os.path.basename(mask)
    mask_vol = nib.load(mask)
    mask_shape = mask_vol.shape
    mask_index = mask_vol.get_data().ravel() > mask_threshold
    n_features = np.sum(mask_index)

    mat_dir = opj(os.path.dirname(sub_path), 'mvp_data')
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    n_feat = len(glob.glob(opj(os.path.dirname(sub_path), '*.feat')))
    n_converted = len(glob.glob(opj(mat_dir, '*header*')))

    # Extract class vector (class_labels)
    class_labels, remove_idx = extract_class_vector(sub_path, remove_class)

    sub_name = os.path.basename(os.path.dirname(sub_path))
    print('Processing %s (run %i / %i)...' % (sub_name, n_converted + 1, n_feat)),

    # Generate and sort paths to stat files (COPEs/tstats)
    if normalize_to_mni:
        stat_dir = opj(sub_path, 'stats')
    else:
        stat_dir = opj(sub_path, 'reg_standard')

    copes = glob.glob(opj(stat_dir, 'cope*.nii.gz'))
    copes = sort_numbered_list(copes)
    _ = [copes.pop(idx) for idx in sorted(remove_idx, reverse=True)]

    varcopes = glob.glob(opj(stat_dir, 'varcope*.nii.gz'))
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
            mvp_data[i_trial, :] = np.divide(mvp_data[i_trial, :], var_sq)

    mvp_data[np.isnan(mvp_data)] = 0

    # Initializing Subject object, which will be saved as a pickle file
    hdr = Subject(class_labels, sub_name, mask_name, mask_index,
                      mask_shape, mask_threshold)

    fn_header = opj(mat_dir, '%s_header_run%i.pickle' % (sub_name, n_converted+1))
    fn_data = opj(mat_dir, '%s_data_run%i.hdf5' % (sub_name, n_converted+1))

    with open(fn_header, 'wb') as handle:
        pickle.dump(hdr, handle)

    h5f = h5py.File(fn_data, 'w')
    h5f.create_dataset('data', data=mvp_data)
    h5f.close()

    print(' done.')

    # Merge if necessary
    if n_feat == (n_converted+1) and n_feat > 1:

        run_headers = glob.glob(opj(mat_dir, '*pickle*'))
        run_data = glob.glob(opj(mat_dir, '*hdf5*'))

        for i in range(len(run_data)):

            if i == 0:
                data = h5py.File(run_data[i], 'r')
                data = data['data']
                hdr = pickle.load(open(run_headers[i], 'rb'))
            else:
                tmp = h5py.File(run_data[i])
                data = np.vstack((data, tmp['data']))
                tmp = pickle.load(open(run_headers[i], 'rb'))
                hdr.class_labels.extend(tmp.class_labels)

        fn_header = opj(mat_dir, '%s_header_merged.pickle' % sub_name)
        fn_data = opj(mat_dir, '%s_data_merged.hdf5' % sub_name)

        with open(fn_header, 'wb') as handle:
            pickle.dump(hdr, handle)

        h5f = h5py.File(fn_data, 'w')
        h5f.create_dataset('data', data=data)
        h5f.close()

