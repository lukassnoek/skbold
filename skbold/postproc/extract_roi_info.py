# Extract region-specific info for a given statistic-file (nifti) and
# outputs a csv-file which can be copy-pasted directly in a Word (ugh..) file.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function
import numpy as np
import nibabel as nib
from glob import glob
import os.path as op
import pandas as pd
from scipy.ndimage.measurements import label

import skbold
roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs', 'harvard_oxford')


def extract_roi_info(statfile, roi_type='unilateral', per_cluster=True,
                     cluster_engine='scipy', min_clust_size=20,
                     stat_threshold=None, mask_threshold=20,
                     save_indices=True):

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

    Returns
    -------
    df : Dataframe
        Dataframe corresponding to the written csv-file.
    """

    data = nib.load(statfile).get_data()
    if stat_threshold:
        data[data < stat_threshold] = 0

    stat_name = op.basename(statfile).split('.')[0]
    masks = glob(op.join(roi_dir, roi_type, '*.nii.gz'))

    df_list = []

    if per_cluster:

        col_names = ['Contrast', 'cluster', 'k cluster', 'max cluster',
                     'x', 'y', 'z', 'Region', 'k region', 'max region']
        results = pd.DataFrame(columns=col_names)

        # Start clustering of data
        if cluster_engine == 'scipy':
            clustered, num_clust = label(data > 0)
            values, counts = np.unique(clustered.ravel(), return_counts=True)
            n_clust = np.argmax(np.sort(counts)[::-1] < min_clust_size)

            if n_clust == 0:
                print('No (sufficiently large) clusters for %s' % statfile)
                return 0

            # Sort and trim
            cluster_nrs = values[counts.argsort()[::-1][:n_clust]]
            cluster_nrs = np.delete(cluster_nrs, 0)
        elif cluster_engine == 'fsl':
            # Not yet implemented
            pass

        print('Analyzing %i clusters for %s' % (len(cluster_nrs), statfile))

        # Looping over clusters
        for i, cluster_id in enumerate(cluster_nrs):
            print('Processing cluster %i' % (i+1))
            cl_mask = clustered == cluster_id
            k = cl_mask.sum()
            mx = data[cl_mask].max()

            tmp = np.zeros(data.shape)
            tmp[cl_mask] = data[cl_mask] == mx
            X = np.where(tmp == 1)[0]
            Y = np.where(tmp == 1)[1]
            Z = np.where(tmp == 1)[2]

            # This is purely for formatting issues
            if i == 0:
                stat = op.basename(statfile).split('.')[0].split('_')[-1]
                c = op.basename(op.dirname(statfile)).split('.')[0] + '_' + stat
            else:
                c = ''

            to_append = {'Contrast': c, 'cluster': (i+1), 'k cluster': k,
                         'max cluster': mx, 'x': X, 'y': Y, 'z': Z,
                         'Region': '', 'k region': '', 'max region': ''}

            df_list.append(pd.DataFrame(to_append, index=[0]))

            # Loop over ROIs
            for mask in masks:
                mask_name = op.basename(mask).split('.')[0].split('_')
                if roi_type == 'unilateral':
                    mask_name[0] += '.'

                mask_name = ' '.join(mask_name)

                roi_mask = nib.load(mask).get_data() > mask_threshold
                overlap = cl_mask.astype(int) + roi_mask.astype(int) == 2
                k = overlap.sum()

                if k > 0:
                    mx = data[overlap].max()
                    tmp = np.zeros(data.shape)
                    tmp[overlap] = data[overlap] == mx
                    X = np.where(tmp == 1)[0]
                    Y = np.where(tmp == 1)[1]
                    Z = np.where(tmp == 1)[2]
                else:
                    # If no voxels, write some default values
                    mx = 0
                    X, Y, Z = 0, 0, 0

                to_append = {'Contrast': '', 'cluster': (i + 1+ 0.1),
                             'k cluster': '', 'max cluster': '', 'x': '',
                             'y': '', 'z': '', 'Region': mask_name,
                             'k region': k, 'max region': mx}

                to_append = pd.DataFrame(to_append, index=[0])
                df_list.append(to_append)

        # Concatenate dataframes!
        df = pd.concat(df_list)

        # Some cleaning / processing
        cols_ordered = ['Contrast', 'cluster', 'k cluster', 'max cluster', 'x',
                        'y', 'z', 'Region', 'k region', 'max region']
        df = df[cols_ordered]
        df = df[df['k region'] > min_clust_size]
        df = df.sort_values(by=['cluster', 'k region'], ascending=[True, False])
        df['cluster'] = ['' if val % 1 != 0 else val for val in df['cluster']]

    else: # If not extracting info per cluster, but wholebrain
        print('Analyzing %s' % statfile)
        col_names = ['roi', 'k', 'max', 'mean', 'sd', 'X','Y', 'Z']
        results = pd.DataFrame(columns=col_names)

        # Only loop over masks
        for mask in masks:
            mask_name = op.basename(mask).split('.')[0].split('_')

            if roi_type == 'unilateral':
                mask_name[0] += '.'
            mask_name = ' '.join(mask_name)

            roi_mask = nib.load(mask).get_data() > mask_threshold
            sig_mask = data > 0
            overlap = sig_mask.astype(int) + roi_mask.astype(int) == 2
            k = overlap.sum()
            X, Y, Z = 0, 0, 0

            if k > 0:
                mx = data[overlap].max()
                mean, std = data[overlap].mean(), data[overlap].std()
                tmp = np.zeros(data.shape)
                tmp[overlap] = data[overlap] == mx
                if save_indices:
                    X = np.where(tmp == 1)[0]
                    Y = np.where(tmp == 1)[1]
                    Z = np.where(tmp == 1)[2]
            else:
                mx, mean, std = 0, 0, 0

            to_append = {'roi': mask_name, 'k': k, 'max': mx, 'mean': mean,
                         'sd': std, 'X': X, 'Y': Y, 'Z': Z}
            to_append = pd.DataFrame(to_append, index=[0])
            df_list.append(to_append)

        df = pd.concat(df_list)
        df = df[df.k != 0].sort_values(by='k', ascending=False)
        df = df[['roi', 'k', 'max', 'mean', 'sd', 'X', 'Y', 'Z']]
        df = df[df['k'] > min_clust_size]

    print(df)
    filename = op.join(op.dirname(statfile), 'roi_info_%s.csv' % stat_name)
    df.to_csv(filename, index=False, header=True, sep='\t')

    return df