# Performs a 'cluster-thresholding' procedure in which clusters smaller
# than a prespecified number are set to zero.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import numpy as np
import nibabel as nib
import os.path as op
from copy import copy
from scipy.ndimage import label


def cluster_size_threshold(data, thresh=None, min_size=20, save=False):
    """ Removes clusters smaller than a prespecified number in a stat-file.

    Parameters
    ----------
    data : numpy-array or str
        3D Numpy-array with statistic-value or a string to a path pointing to
        a nifti-file with statistic values.
    thresh : int, float
        Initial threshold to binarize the image and extract clusters.
    min_size : int
        Minimum size (i.e. amount of voxels) of cluster. Any cluster with fewer
        voxels than this amount is set to zero ('removed').
    save : bool
        If data is a file-path, this parameter determines whether the cluster-
        corrected file is saved to disk again.
    """

    if isinstance(data, (str, unicode)):
        fname = copy(data)
        data = nib.load(data)
        affine = data.affine
        data = data.get_data()

    if thresh is not None:
        data[data < thresh] = 0

    clustered, num_clust = label(data > 0)
    values, counts = np.unique(clustered.ravel(), return_counts=True)

    # Get number of clusters by finding the index of the first instance
    # when 'counts' is smaller than min_size
    first_clust = np.sort(counts)[::-1] < min_size
    if first_clust.sum() == 0:
        print('All clusters were larger than: %i, returning original data' %
              min_size)
        return data

    n_clust = np.argmax(first_clust)

    # Sort and trim
    cluster_nrs = values[counts.argsort()[::-1][:n_clust]]
    cluster_nrs = np.delete(cluster_nrs, 0)

    # Set small clusters to zero.
    data[np.invert(np.in1d(clustered, cluster_nrs)).reshape(data.shape)] = 0

    if save:
        img = nib.Nifti1Image(data, affine=affine)
        basename = op.basename(fname)
        nib.save(img, basename.split('.')[0] + '_thresholded.nii.gz')

    return data
