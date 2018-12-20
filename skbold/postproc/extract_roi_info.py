# Extract region-specific info for a given statistic-file (nifti) and
# outputs a csv-file which can be copy-pasted directly in a Word (ugh..) file.

# Note: this is quite likely the ugliest code I've ever written.
# My apologies.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function, absolute_import
from builtins import range
import numpy as np
import nibabel as nib
from glob import glob
import os.path as op
import pandas as pd
from scipy.ndimage.measurements import label
from nilearn.datasets import fetch_atlas_harvard_oxford, load_mni152_template
from nilearn.image import coord_transform


def extract_roi_info(statfile, stat_name=None, roi_type='unilateral',
                     per_cluster=True, cluster_engine='scipy',
                     min_clust_size=20, stat_threshold=None, mask_threshold=20,
                     save_indices=True, verbose=True):

    """
    Extracts information per ROI for a given statistics-file.
    Reads in a thresholded (!) statistics-file (such as a thresholded z- or
    t-stat from a FSL first-level directory) and calculates for a set of ROIs
    the number of significant voxels included and its maximum value
    (+ coordinates). Saves a csv-file in the same directory as the
    statistics-file. Assumes that the statistics file is in MNI152 2mm space.

    Parameters
    ----------
    statfile : str
        Absolute path to statistics-file (nifti) that needs to be evaluated.
    stat_name : str
        Name for the contrast/stat-file that is being analyzed.
    roi_type : str
        Whether to  use unilateral or bilateral masks (thus far, only Harvard-
        Oxford atlas masks are supported.)
    per_cluster : bool
        Whether to evaluate the statistics-file as a whole (per_cluster=False)
        or per cluster separately (per_cluster=True).
    cluster_engine : str
        Which 'engine' to use for clustering; can be 'scipy' (default), using
        scipy.ndimage.measurements.label, or 'fsl' (using FSL's cluster
        commmand).
    min_clust_size : int
        Minimum cluster size (i.e. clusters with fewer voxels than this number
        are discarded; also, ROIs containing fewer voxels than this will not
        be listed on the CSV.
    stat_threshold : int or float
        If the stat-file contains uncorrected data, stat_threshold can be used
        to set a lower bound.
    mask_threshold : bool
        Threshold for probabilistics masks, such as the Harvard-Oxford masks.
        Default of 25 is chosen as this minimizes overlap between adjacent
        masks while still covering most of the entire brain.
    save_indices : bool
        Whether to save the indices (coordinates) of peaks of clusters.
    verbose : bool
        Whether to print some output regarding the parsing process.

    Returns
    -------
    df : Dataframe
        Dataframe corresponding to the written csv-file.
    """

    if isinstance(statfile, str):
        data = nib.load(statfile).get_data()
    else:
        data = statfile

    sign_mask = np.ones(shape=data.shape)
    sign_mask[data < 0] = -1

    mni_affine = load_mni152_template().affine

    if stat_threshold:
        data[data < stat_threshold] = 0

    if roi_type == 'unilateral':
        cort_rois = fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split=True)
        subc_rois = fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm', symmetric_split=True)
    else:
        cort_rois = fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split=True)
        subc_rois = fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm', symmetric_split=True)

    cort_rois['idx'] = np.arange(len(cort_rois['labels']))
    subc_rois['idx'] = np.arange(len(subc_rois['labels']))

    df_list = []

    # Start clustering of data
    clustered, _ = label(data > 0)
    values, counts = np.unique(clustered.ravel(), return_counts=True)
    n_clust = np.argmax(np.sort(counts)[::-1] < min_clust_size)

    if n_clust == 0:
        print('No (sufficiently large) clusters!')
        return 0

    # Sort and trim
    cluster_nrs = values[counts.argsort()[::-1][:n_clust]]
    cluster_nrs = np.delete(cluster_nrs, 0)

    print('Analyzing %i clusters for %s' % (len(cluster_nrs), stat_name))

    cluster_list = []
    # Looping over clusters
    for i, cluster_id in enumerate(cluster_nrs):

        if verbose:
            print('Processing cluster %i' % (i+1))

        cl_mask = clustered == cluster_id
        k = cl_mask.sum()
        mx = data[cl_mask].max()

        tmp = np.zeros(data.shape)
        tmp[cl_mask] = data[cl_mask] == mx

        # in case of multiple voxels with same stat / weight
        if np.sum(tmp == 1) > 1:
            X, Y, Z = [coord[0] for coord in np.where(tmp == 1)]
        else:
            X, Y, Z = np.where(tmp == 1)

        # if weight / stat is negative, change sign of mx
        if sign_mask[X, Y, Z] < 0:
            mx = -mx

        # convert to MNI-coordinates
        X, Y, Z = coord_transform(X, Y, Z, mni_affine)

        # This is purely for formatting issues
        if i == 0:
            c = stat_name   
        else:
            c = ''

        to_append = {'Contrast': c, 'cluster': (i+1), 'k cluster': str(k),
                        'max cluster': mx, 'x': str(X[0]), 'y': str(Y[0]),
                        'z': str(Z[0]),
                        'Region': '', 'k region': '', 'max region': ''}

        cluster_list.append(to_append)

        roi_list = []
        # Loop over ROIs
        for i_atlas, atlas in enumerate([cort_rois, subc_rois]):
            maps = atlas['maps'].get_data()

            for ii_roi, mask_name in enumerate(atlas['labels']):
                roi_mask = maps == atlas['idx'][ii_roi]
                overlap = (cl_mask.astype(int) + roi_mask.astype(int)) == 2
                k = overlap.sum()
                
                if k > 0:
                    mx = data[overlap].max()
                    tmp = np.zeros(data.shape)
                    tmp[overlap] = data[overlap] == mx

                    # in case of multiple voxels with same stat / weight
                    if np.sum(tmp == 1) > 1:
                        X, Y, Z = [coord[0] for coord in np.where(tmp == 1)]
                    else:
                        X, Y, Z = np.where(tmp == 1)

                    # if sign of weight / stat is negative, change sign of mx
                    if sign_mask[X, Y, Z] < 0:
                        mx = -mx

                    # convert to MNI-coordinates
                    X, Y, Z = coord_transform(X, Y, Z, mni_affine)

                else:
                    # If no voxels, write some default values
                    mx = 0
                    X, Y, Z = 0, 0, 0
                
                if ii_roi == 0:
                    cluster_list[-1]['Region'] = mask_name
                    cluster_list[-1]['k region'] = k
                    cluster_list[-1]['max region'] = mx
                else:
                    
                    to_append = {'Contrast': '', 'cluster': (i + 1 + 0.1),
                                'k cluster': '', 'max cluster': '', 'x': '',
                                'y': '', 'z': '', 'Region': mask_name,
                                'k region': k, 'max region': mx}
                    
                    if k > cluster_list[-1]['k region']:
                        to_append['Region'] = cluster_list[-1]['Region']
                        to_append['k region'] = cluster_list[-1]['k region']
                        maxr = cluster_list[-1]['max region']
                        to_append['max region'] = maxr
                        cluster_list[-1]['Region'] = mask_name
                        cluster_list[-1]['k region'] = k
                        cluster_list[-1]['max region'] = mx

                    roi_list.append(to_append)

            roi_dfs = pd.concat([pd.DataFrame(tmp, [0]) for tmp in roi_list])
            roi_dfs = roi_dfs[roi_dfs['k region'] > min_clust_size]
            roi_dfs = roi_dfs.sort_values(by=['cluster', 'k region'],
                                          ascending=[True, False])

            whiteline = {key: '' if key != 'cluster' else (i + 1 + 0.1)
                         for key in roi_dfs.keys()}

            roi_dfs = roi_dfs.append(whiteline, ignore_index=True)
            df_list.append(pd.DataFrame(cluster_list[-1], [0]))
            df_list.append(roi_dfs)

        # Concatenate dataframes!
        df = pd.concat(df_list)
        # Some cleaning / processing
        cols_ordered = ['Contrast', 'cluster', 'k cluster', 'max cluster', 'x',
                        'y', 'z', 'Region', 'k region', 'max region']
        df = df[cols_ordered]

        df['cluster'] = ['' if (val % 1) != 0 else str(val)
                         for val in df['cluster']]

    filename = op.join(op.dirname(statfile), 'roi_info_%s.csv' % stat_name)
    df.to_csv(filename, index=False, header=True, sep='\t')

    return df
