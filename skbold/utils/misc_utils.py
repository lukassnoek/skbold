"""
Utility functions/classes unrelated to multivoxel pattern analysis.
"""

from __future__ import division, print_function
import numpy as np
import nibabel as nib
import os
from glob import glob
import os.path as op
import pandas as pd
import skbold.ROIs.harvard_oxford as roi
from scipy.ndimage.measurements import label


def extract_roi_info(statfile, roi_type='unilateral', per_cluster=True,
                     threshold=30):
    """ Extracts information per ROI for a given statistics-file.

    Reads in a thresholded (!) statistics-file (such as a thresholded z- or
    t-stat from a FSL first-level directory) and calculates for a set of ROIs
    the number of significant voxels included and its maximum value (+ coordinates).
    Saves a csv-file in the same directory as the statistics-file.
    Assumes that the statistics file is in MNI152 2mm space. 

    Parameters
    ----------
    statfile : str
        Absolute path to statistics-file (nifti) that needs to be evaluated.
    roi_type : str
        Whether to  use unilateral or bilateral masks (thus far, only Harvard-
        Oxford atlas masks are supported.)
    per_cluster : bool
        Whether to evaluate the statistics-file as a whole (per_cluster=False) or
        per cluster separately (per_cluster=True).
    threshold : bool
        Threshold for probabilistics masks, such as the Harvard-Oxford masks.

    """

    data = nib.load(statfile).get_data()
    stat_name = op.basename(statfile).split('.')[0]
    roi_dir = op.join(op.dirname(roi.__file__), roi_type)
    masks = glob(op.join(roi_dir, '*.nii.gz'))

    df_list = []

    if per_cluster:

        results = pd.DataFrame(columns=['cluster', 'roi', 'k', 'max', 'X', 'Y', 'Z'])
        clustered, num_clust = label(data > 0)
        values, counts = np.unique(clustered.ravel(), return_counts=True)
        n_clust = np.argmax(np.sort(counts)[::-1] < 20)

        # Sort and trim
        cluster_nrs = values[counts.argsort()[::-1][:n_clust]]
        cluster_nrs = np.delete(cluster_nrs, 0)

        print('Analyzing %i clusters for %s' % (len(cluster_nrs), statfile))

        for i, cluster_id in enumerate(cluster_nrs):
            print('Processing cluster %i' % (i+1))
            cl_mask = clustered == cluster_id
            k = cl_mask.sum()
            mx = data[cl_mask].max()

            tmp = np.zeros(data.shape)
            tmp[cl_mask] = data[cl_mask] == mx
            X, Y, Z = np.where(tmp == 1)[0], np.where(tmp == 1)[1], np.where(tmp == 1)[2]
            
            to_append = {'cluster': (i+1), 'roi': 'full cluster',
                         'k': k, 'max': mx, 'X': X, 'Y': Y, 'Z': Z}

            df_list.append(pd.DataFrame(to_append, index=[0]))

            for mask in masks:
                mask_name = op.basename(mask).split('.')[0].split('_')
                mask_name[0] = mask_name[0] + '.'
                mask_name = " ".join(mask_name)

                roi_mask = nib.load(mask).get_data() > threshold
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
                    mx = 0
                    X, Y, Z = 0, 0, 0

                to_append = pd.DataFrame({'cluster': (i+1), 'roi': mask_name,
                         'k': k, 'max': mx, 'X': X, 'Y': Y, 'Z': Z}, index=[0])
                df_list.append(to_append)
        df = pd.concat(df_list)
        df = df[['cluster', 'roi', 'k', 'max', 'X', 'Y', 'Z']]  
        df = df[df.k != 0].sort_values(by=['cluster', 'k'], ascending=[True,False])

    else:
        print('Analyzing %s' % statfile)
        results = pd.DataFrame(columns=['roi', 'k', 'max', 'X', 'Y', 'Z'])

        for mask in masks:
                mask_name = op.basename(mask).split('.')[0].split('_')
                mask_name[0] = mask_name[0] + '.'
                mask_name = " ".join(mask_name)

                roi_mask = nib.load(mask).get_data() > threshold
                sig_mask = data > 0
                overlap = sig_mask.astype(int) + roi_mask.astype(int) == 2
                k = overlap.sum()

                if k > 0:
                    mx = data[overlap].max()
                    tmp = np.zeros(data.shape)
                    tmp[overlap] = data[overlap] == mx
                    X = np.where(tmp == 1)[0]
                    Y = np.where(tmp == 1)[1]
                    Z = np.where(tmp == 1)[2]
                else:
                    mx = 0
                    X, Y, Z = 0, 0, 0


                to_append = pd.DataFrame({'roi': mask_name, 'k': k, 'max': mx,
                             'X': X, 'Y': Y, 'Z': Z}, index=[0])
                df_list.append(to_append)
        df = pd.concat(df_list)
        df = df[df.k != 0].sort_values(by='k', ascending=False)
        df = df[['roi', 'k', 'max', 'X', 'Y', 'Z']]
    
    print(df)
    filename = op.join(op.dirname(statfile), 'roi_info_%s.csv' % stat_name)
    df.to_csv(filename, index=False, header=True, sep='\t')


if __name__ == '__main__':
    "Just testing stuff"
    testfile = '/media/lukas/data/DecodingEmotions/univar_zinnen/Cope4.gfeat/thresh_zstat1.nii.gz'
    extract_roi_info(testfile, roi_type='unilateral', per_cluster=True,
                     threshold=30)