# Class to index a whole-brain pattern with a certain ROI.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
import os
import glob
import os.path as op
import nibabel as nib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage.measurements import label


class MultiRoiIndexer(BaseEstimator, TransformerMixin):
    """ Wrapper that calls RoiIndexer multiple times for Fsl2mvpBetween mvps.

    Applies masks of multiple regions of interest.
    ROIs can be either anatomical (from a probabilistic map), or functional
    (z-stats obtained from a group-level analysis). This method also
    creates cluster ids: it finds clusters of contiguous voxels. Probably
    not so useful for anatomical ROIs but very useful for functional ROIs.

    Parameters
    ----------
    mvp : mvp-object (see scikit_bold.core)
        Mvp-object, necessary to extract some pattern metadata
    maskdict : dict of dicts
        dictionary with KEYS = COPE-names as they occur in self.X_dict;
        VALUE = dict with KEYS 'path' (absolute path to mask *.nii.gz file)
        and 'threshold' (threshold to be applied to that path)
    """

    def __init__(self, mvp, maskdict, verbose=False):

        self.mvp = mvp
        self.maskdict = maskdict
        self.orig_mask = mvp.mask_index
        self.directory = mvp.directory
        self.ref_space = mvp.ref_space
        self.idx_ = None
        self.verbose = verbose

    def fit(self, X=None, y=None):
        """ Fits MultiRoiIndexer.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : List of str
            List or ndarray with floats corresponding to labels
        """

        contrast_labels = self.mvp.contrast_labels
        maskdict = self.maskdict

        #initialize some vars
        roi_idx = np.ones(0, dtype=bool)
        cluster_id = np.ones(0, dtype=np.uint8)
        total_clusters = 0

        for copeindex, cope in enumerate(contrast_labels):
            if self.verbose:
                print('Cope: %s, path: %s, threshold: %f' %(cope, maskdict[cope]['path'], maskdict[cope]['threshold']))

            #Load mask and threshold
            roi_idx_cope = nib.load(maskdict[cope]['path']).get_data() > maskdict[cope]['threshold']

            #Identify clusters: use MNI full-brain space (including white matter)
            cluster_id_ccope, n_clusters_ccope = label(roi_idx_cope)
            cluster_id_ccope = cluster_id_ccope.ravel()
            cluster_id_ccope[cluster_id_ccope>0] += total_clusters

            #Update total number of clusters
            total_clusters = total_clusters + n_clusters_ccope

            #Apply original mask (e.g., gray matter) to new mask (e.g., ROI).
            overlap = roi_idx_cope.astype(int).ravel() + self.orig_mask.astype(int)
            roi_idx_thiscope = (overlap==2)[self.orig_mask]

            #Apply original mask to cluster_id_ccope (e.g., only index gray matter)
            cluster_id_ccope = cluster_id_ccope[self.orig_mask]

            #Use new mask to index in cluster labels (e.g., only index ROI)
            cluster_id_ccope = cluster_id_ccope[roi_idx_thiscope]

            #Concatenate old and new roi_idx and cluster_labels
            roi_idx = np.hstack([roi_idx, roi_idx_thiscope])
            cluster_id = np.hstack([cluster_id, cluster_id_ccope])
            if self.verbose:
                print('Current cope loaded, size: %f' %(roi_idx_thiscope.size))
                print('Cluster idx in current cope: ')
                print(cluster_id)
                print('\nSize of total roi_idx: %f' %(roi_idx[roi_idx==True]).size)
                print('Size of total cluster_id: %f' %cluster_id.size)

        self.idx_ = roi_idx
        self.mvp.cluster_id = cluster_id
        self.mvp.contrast_id = self.mvp.contrast_id[roi_idx]
        self.mvp.voxel_idx = self.mvp.voxel_idx[roi_idx]

        return self

    def transform(self, X, y=None):
        """ Transforms features from X (voxels) to a mask-subset.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        y : Optional[List[str] or numpy ndarray[str]]
            List of ndarray with strings indicating label-names

        Returns
        -------
        X_new : ndarray
            array with transformed data of shape = [n_samples, n_features]
            in which features are region-average values.
        """

        X_new = X[:, self.idx_]

        return X_new