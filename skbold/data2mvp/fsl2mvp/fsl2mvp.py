# Class to extract and store first-level (meta)data from an FSL first-level
# (feat) directory.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
import cPickle
import h5py
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import os
import glob
import os.path as op
from skbold.utils import sort_numbered_list
from skbold.core import Mvp
from sklearn.preprocessing import LabelEncoder


class Fsl2mvp(Mvp):
    """ Fsl2mvp (multiVoxel Pattern) class, a subclass of Mvp (skbold.core)

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).
    """

    def __init__(self, directory, mask_threshold=0, beta2tstat=True,
                 ref_space='epi', mask_path=None, remove_class=[]):

        super(Fsl2mvp, self).__init__(directory, mask_threshold, beta2tstat,
                                      ref_space, mask_path, remove_class)

    def _read_design(self):
        design_file = op.join(self.directory, 'design.con')

        if not os.path.isfile(design_file):
            raise IOError('There is no design.con file for %s' % self.sub_name)

        # Find number of contrasts and read in accordingly
        contrasts = sum(1 if 'ContrastName' in line else 0
                        for line in open(design_file))

        n_lines = sum(1 for line in open(design_file))

        df = pd.read_csv(design_file, delimiter='\t', header=None,
                         skipfooter=n_lines-contrasts, engine='python')

        class_labels = list(df[1])
        return class_labels

    def _extract_class_labels(self):
        """ Extracts class labels as strings from FSL first-level directory.

        This method reads in a design.con file, which is by default outputted
        in an FSL first-level directory, and sets self.class_labels to a list
        with labels, and in addition sets self.remove_idx with indices which
        trials (contrasts) were removed as indicated by the remove_class
        attribute from the Fsl2mvp object.

        """

        remove_class = self.remove_class
        class_labels = self._read_design()

        # Remove to-be-ignored contrasts (e.g. cues)
        remove_idx = np.zeros((len(class_labels), len(remove_class)))

        for i, name in enumerate(remove_class):
            remove_idx[:, i] = np.array([name in label for label in class_labels])

        self.remove_idx = np.where(remove_idx.sum(axis=1).astype(int))[0]
        _ = [class_labels.pop(idx) for idx in np.sort(self.remove_idx)[::-1]]

        # Here, numeric extensions of contrast names (e.g. 'positive_003') are
        # removed
        self.class_labels = []
        for c in class_labels:
            parts = c.split('_')
            parts = [x.strip() for x in parts]
            if parts[-1].isdigit():
                label = '_'.join(parts[:-1])
                self.class_labels.append(label)
            else:
                self.class_labels.append(c)