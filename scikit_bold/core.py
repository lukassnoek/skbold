# -*- coding: utf-8 -*-
""" Core module

This module contains some core classes/functions which are central to the
scikit-bold package. Most importantly, it contains the Mvp class, an object
meant for storage of multivoxel pattern data and its associated meta-data.

In the data2mvp module, several classes subclass the Mvp class, adding
methods specific to that class (e.g. the fslglm object subclasses Mvp and
adds methods to extract first-level single-trial data from fsl-specific
fist-level directories.)

Lukas Snoek
"""

from __future__ import print_function, absolute_import, division
import glob
import cPickle
import h5py
import numpy as np
import os.path as op
from sklearn.preprocessing import LabelEncoder


class Mvp(object):
    """ Mvp (multiVoxel Pattern) class.

    Creates an object, specialized for storing fMRI data that will be analyzed
    using machine learning or RSA-like analyses, that stores both the data
    (X: an array of samples by features, y: numeric labels corresponding to
    X's classes/conditions) and the corresponding meta-data (e.g. nifti header,
    mask info, etc.).

    """

    def __init__(self, directory, mask_threshold=0, beta2tstat=True,
                 ref_space='mni', mask_path=None, remove_class=[],
                 cleanup=True):
        """ Initializes a (bare-bones) Mvp object.

        Parameters
        ----------
        directory : str
            Absolute path to directory from which first-level data should be
            extracted.
        mask_threshold : Optional[int or float]
            If a probabilistic mask is used, mask_threshold sets the lower-
            bound for the mask
        beta2tstat : bool
            Whether to convert extracted beta-values to t-statistics by
            dividing by their corresponding standard deviation.
        ref_space : str
            Indicates in which space the multivoxel patterns should be
            returned, either 'mni' (MNI152 2mm space) or 'epi' (native
            functional space). Thus far, MNI space only works for first-level
            data returned by fsl.
        mask_path : str
            Absolute path to the mask that will be used to index the patterns
            with.
        remove_class : list[str]
            List of condition names (or substrings of condition names) that
            need not to be included in the pattern-data (e.g. covariates,
            nuisance regressors, etc.).

        """

        self.directory = directory
        self.sub_name = op.basename(op.dirname(directory))
        self.run_name = op.basename(directory).split('.')[0].split('_')[-1]
        self.ref_space = ref_space
        self.beta2tstat = beta2tstat
        self.mask_path = mask_path
        self.mask_threshold = mask_threshold

        if mask_path is not None:
            self.mask_name = op.basename(op.dirname(mask_path))
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

    def update_metadata(self):
        # Maybe change this to work with @property and setters
        cl = self.class_labels
        self.y = LabelEncoder().fit_transform(cl)
        self.n_trials = len(cl)
        self.class_names = np.unique(cl)
        self.n_class = len(self.class_names)
        self.n_inst = [np.sum(cls == cl) for cls in cl]
        self.class_idx = [cl == cls for cls in self.class_names]
        self.trial_idx = [np.where(cl == cls)[0] for cls in self.class_names]

    def merge_runs(self, cleanup=False, iD='merged'):
        """ Merges single-trial patterns from different runs.

        Given two runs, this method merges their single-trial patterns by
        simple concatenation; importantly, it assumes that the runs are
        identical in their set-up (e.g. conditions).

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
        run_headers = glob.glob(op.join(mat_dir, '*pickle*'))
        run_data = glob.glob(op.join(mat_dir, '*hdf5*'))

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
                    # Concatenate data to first run and extend class_labels
                    tmp = h5py.File(run_data[i])
                    data = np.vstack((data, tmp['data'][:]))
                    tmp.close()
                    tmp = cPickle.load(open(run_headers[i], 'r'))
                    hdr.class_labels.extend(tmp.class_labels)

            hdr.update_metadata()
            hdr.y = LabelEncoder().fit_transform(hdr.class_labels)
            fn_header = op.join(mat_dir, '%s_header_%s.pickle' %
                                (self.sub_name, iD))
            fn_data = op.join(mat_dir, '%s_data_%s.hdf5' %
                              (self.sub_name, iD))

            with open(fn_header, 'wb') as handle:
                cPickle.dump(hdr, handle)

            h5f = h5py.File(fn_data, 'w')
            h5f.create_dataset('data', data=data)
            h5f.close()
        else:
            pass
