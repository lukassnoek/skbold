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
from glob import glob
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
                 ref_space='epi', mask_path=None, remove_contrast=[],
                 invert_selection=False):
        """

        Parameters
        ----------
        directory : feat directory of subject
        mask_threshold : threshold for masking
        beta2tstat : convert beta to t-stat?
        ref_space : 'mni' or 'epi'
        mask_path : path to mask (can be None)
        remove_class : which contrasts to remove
        invert_selection : if True, remove_class = which contrasts to include
        """

        super(Fsl2mvp, self).__init__(mask_threshold, beta2tstat, ref_space, mask_path)
        self.directory = directory

        if not op.exists(directory):
            raise OSError("The directory '%s' doesn't seem to exist!" % directory)
        self.beta2tstat = beta2tstat
        self.sub_name = op.basename(op.dirname(directory))
        self.run_name = op.basename(self.directory).split('.')[0].split('_')[-1]
        self.remove_contrast = remove_contrast
        self.invert_selection = invert_selection
        self.contrast_labels = None
        self.voxel_idx = np.zeros(0, dtype=np.uint32)

    def _read_design(self):
        design_file = op.join(self.directory, 'design.con')

        if not op.isfile(design_file):
            raise IOError('There is no design.con file for %s' % design_file)

        # Find number of contrasts and read in accordingly
        contrasts = sum(1 if 'ContrastName' in line else 0
                        for line in open(design_file))

        n_lines = sum(1 for line in open(design_file))

        df = pd.read_csv(design_file, delimiter='\t', header=None,
                         skipfooter=n_lines - contrasts, engine='python')

        cope_labels = list(df[1].str.strip()) #remove spaces
        return cope_labels

    def _extract_labels(self):
        """ Extracts class labels as strings from FSL first-level directory.

        This method reads in a design.con file, which is by default outputted
        in an FSL first-level directory, and sets self.class_labels to a list
        with labels, and in addition sets self.remove_idx with indices which
        trials (contrasts) were removed as indicated by the remove_class
        attribute from the Fsl2mvp object.
        """

        cope_labels = self._read_design()
        remove_contrast = self.remove_contrast

        # Remove to-be-ignored contrasts (e.g. cues)
        remove_idx = np.zeros((len(cope_labels), len(remove_contrast)))

        for i, name in enumerate(remove_contrast):
            remove_idx[:, i] = np.array([name in label for label in cope_labels])

        self.remove_idx = np.where(remove_idx.sum(axis=1).astype(int))[0]

        if self.invert_selection:
            self.remove_idx = [x for x in np.arange(len(cope_labels)) if not x in self.remove_idx]

        _ = [cope_labels.pop(idx) for idx in np.sort(self.remove_idx)[::-1]]

        # Here, numeric extensions of contrast names
        # (e.g. 'positive_003') are removed
        labels = []
        for c in cope_labels:
            parts = [x.strip() for x in c.split('_')]
            if parts[-1].isdigit():
                label = '_'.join(parts[:-1])
                labels.append(label)
            else:
                labels.append(c)

        self.contrast_labels = labels

    def merge_runs(self, idf='merged', cleanup=True):

        """ Merges single-trial patterns from different runs.

          Given m runs, this method merges patterns by simple concatenation.
          Concatenation either occurs along the horizontal axis (if the design
          is between subjects) or along the vertical axis (if the design is within
          subjects). Importantly, for within subject designs, it assumes that
          runs are identical in their set-up (e.g., conditions).

          Parameters
          ----------
          cleanup : bool
              Whether to clean up the run-wise data and thus to keep only the
              merged data.
          id : str
              Identifier to give the merged runs, such that the data and header
              files have the structure of: <subname>_header/data_<id>.extension
          """

        mat_dir = op.join(op.dirname(self.directory), 'mvp_data')
        run_headers = glob(op.join(mat_dir, '*pickle*'))
        run_data = glob(op.join(mat_dir, '*hdf5*'))

        if len(run_headers) > 1:
            print('Merging runs for %s' % self.sub_name)

            for i in range(len(run_data)):

                # 'Peek' at first run
                if i == 0:
                    h5f = h5py.File(run_data[i], 'r')
                    data = h5f['data'][:]
                    h5f.close()
                    hdr = cPickle.load(open(run_headers[i]))
                else:
                    # Concatenate data to first run and extend cope_labels
                    tmp = h5py.File(run_data[i])

#                    if self.__class__.__name__ == 'Fsl2mvpWithin':
#                    data = np.concatenate((data, tmp['data'][:]), axis=0)
#                    elif self.__class__.__name__ == 'Fsl2mvpBetween':
                    tmpdat = tmp['data'][:]
                    data = np.concatenate((data, tmpdat), axis=0)
                    tmp.close()

                    tmp = cPickle.load(open(run_headers[i], 'r'))
                    hdr.contrast_labels.extend(tmp.contrast_labels)

                    if self.__class__.__name__ == 'Fsl2mvpBetween':
                        to_concat = tmp.contrast_id + len(np.unique(hdr.contrast_id))
                        hdr.contrast_id = np.concatenate((hdr.contrast_id, to_concat), axis=0)
                        hdr.voxel_idx = np.concatenate((hdr.voxel_idx, tmp.voxel_idx))
                        # hdr._update_X_dict(tmp.X_dict)

                if self.__class__.__name__ == 'Fsl2mvpWithin':
                    hdr.y = LabelEncoder().fit_transform(hdr.contrast_labels)

            hdr._update_metadata()

            fn_header = op.join(mat_dir, '%s_header_%s.pickle' %
                                (self.sub_name, idf))
            fn_data = op.join(mat_dir, '%s_data_%s.hdf5' %
                              (self.sub_name, idf))

            with open(fn_header, 'wb') as handle:
                cPickle.dump(hdr, handle)

            hdr.X = data
            h5f = h5py.File(fn_data, 'w')
            h5f.create_dataset('data', data=data)
            h5f.close()

            if cleanup:
                run_headers.extend(run_data)
                _ = [os.remove(f) for f in run_headers]
        else:
            # If there's only one file, don't merge
            pass

        return hdr

