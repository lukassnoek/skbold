from __future__ import print_function, division, absolute_import
import numpy as np
import glob
import os
import cPickle
import h5py
import json
import pandas as pd
import os.path as op
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
     confusion_matrix
from skbold.transformers import *
from nipype.interfaces import fsl
from itertools import chain, combinations
from scipy.misc import comb
import nibabel as nib


class DataHandler(object):
    """ Loads in data and headers and merges them in a Mvp object.

    Loads in data and metadata of multivoxel patterns in a multitude of ways,
    including within-subject single trials (load_separate_sub), between-
    subjects single trials (load_concatenated_subs), between-subjects
    averaged trials (i.e. preserves order of trials, in which trials of the
    same presentation order are averaged across subjects), and between-subjects
    contrasts (i.e. trials from the same condition are averaged within subjects
    and are used as samples).

    """
    def __init__(self, identifier='', shape='2D'):
        """ Initializes DataHandler object.

        Parameters
        ----------
        identifier : str
            Identifier which should be included in the data/header names for
            them to be loaded in.
        shape : str
            Indicates which shape the multivoxel pattern(s) should have (can be
                either '2D', as usual for scikit-learn style analyses or '4D',
                which is necessary for, e.g., searchlight-based analyses in
                nilearn).
        """
        self.identifier = identifier
        self.shape = shape
        self.mvp = None

    def load_separate_sub(self, sub_dir):
        """ Loads the (meta)data from a single subject.

        Parameters
        ----------
        sub_dir : str
            Absolute path to a subject directory, assuming that it contains
            a mvp_data directory.

        Returns
        -------
        mvp : Mvp object (see scikit_bold.core module)

        """
        mvp_dir = op.join(sub_dir, 'mvp_data')
        data_path = glob.glob(op.join(mvp_dir, '*%s*.hdf5' % self.identifier))
        hdr_path = glob.glob(op.join(mvp_dir, '*%s*.pickle' % self.identifier))

        if len(data_path) > 1 or len(hdr_path) > 1:
            raise ValueError('Try to load more than one data/hdr file ...')
        elif len(data_path) == 0 or len(hdr_path) == 0:
            raise ValueError('No data and/or header paths found!')

        mvp = cPickle.load(open(hdr_path[0]))
        h5f = h5py.File(data_path[0], 'r')
        mvp.X = h5f['data'][:]
        h5f.close()

        if self.shape == '4D':
            s = mvp.mask_shape

            # This is a really ugly hack, but for some reason the following
            # doesn't work: mvp.X.reshape((s[0],s[1],s[2],mvp.X.shape[0]))
            tmp_X = np.zeros((s[0], s[1], s[2], mvp.X.shape[0]))
            for trial in range(mvp.X.shape[0]):
                tmp_X[:, :, :, trial] = mvp.X[trial, :].reshape(mvp.mask_shape)
            mvp.X = tmp_X

        self.mvp = mvp

        return mvp

    def load_concatenated_subs(self, directory):
        """ Loads single-trials from multiple subjects and concatenates them.

        Given a directory with subject-specific subdirectories, this method
        load the single-trial (meta)data from each subject and subsequently
        concatenates these patterns such that the resulting Mvp object contains
        data of shape = [n_trials * n_subjects, n_features].

        Parameters
        ----------
        directory : str
            Absolute path to directory containing subject-specific
            subdirectories, each containing a mvp_data directory.

        Returns
        -------
        mvp : Mvp object (see scikit_bold.core module)

        """
        iD = self.identifier
        data_name = op.join(directory, '*', 'mvp_data', '*%s*.hdf5' % iD)
        hdr_name = op.join(directory, '*', 'mvp_data', '*%s*.pickle' % iD)
        data_paths, hdr_paths = glob.glob(data_name), glob.glob(hdr_name)

        # Peek at first
        for i in range(len(data_paths)):

            if i == 0:
                h5f = h5py.File(data_paths[i], 'r')
                data = h5f['data'][:]
                h5f.close()
                mvp = cPickle.load(open(hdr_paths[i]))

                if mvp.ref_space == 'epi':
                    msg = 'Cannot concatenate subs from different epi spaces!'
                    raise ValueError(msg)

            else:
                tmp = h5py.File(data_paths[i])
                data = np.vstack((data, tmp['data'][:]))
                tmp.close()
                tmp = cPickle.load(open(hdr_paths[i], 'r'))
                mvp.class_labels.extend(tmp.class_labels)

        mvp.update_metadata()
        mvp.X = data
        mvp.sub_name = 'ConcatenatedSubjects'
        self.mvp = mvp

        return mvp

    def load_averaged_subs(self, directory):
        """ Loads single-trial within-subject data and averages them.

        This method loads in single-trial data from separate subjects and
        averages each trial across subjects. The order of the trials should
        make sense per subject (is not guaranteed right now.).

        Parameters
        ----------
        directory : str
            Absolute path to directory containing subject-specific
            subdirectories, each containing a mvp_data directory.

        Returns
        -------
        mvp : Mvp object (see scikit_bold.core module)

        """

        iD = self.identfier
        data_name = op.join(directory, '*', 'mvp_data', '*%s*.hdf5' % iD)
        hdr_name = op.join(directory, '*', 'mvp_data', '*%s*.pickle' % iD)
        data_paths, hdr_paths = glob.glob(data_name), glob.glob(hdr_name)

        # Peek at first
        for i in range(len(data_paths)):
            if i == 0:
                h5f = h5py.File(data_paths[i], 'r')
                data_tmp = h5f['data'][:]
                h5f.close()
                mvp = cPickle.load(open(hdr_paths[i]))

                if mvp.ref_space == 'epi':
                    msg = 'Cannot concatenate subs from different epi spaces!'
                    raise ValueError(msg)

                # Pre-allocation
                shape = data_tmp.shape
                data = np.zeros((len(data_paths), shape[0], shape[1]))
                data[i, :, :] = data_tmp

            else:
                tmp = h5py.File(data_paths[i])
                data[i, :, :] = tmp['data'][:]
                tmp.close()

        mvp.X = data.mean(axis=0)
        mvp.sub_name = 'AveragedSubjects'
        self.mvp = mvp
        return mvp

    def load_averagedcontrast_subs(self, directory, grouping):
        """ Loads single-trials and averages trials within conditions.

        Loads single-trials but averages within conditions (given a certain
        grouping, indicating conditions within factorial designs) and
        subsequently concatenates these 'univariate' patterns across subjects.

        Parameters
        ----------
        directory : str
            Absolute path to directory containing subject-specific
            subdirectories, each containing a mvp_data directory.
        grouping : list[str]
            Indication of a factorial 'grouping' over which trials should be
            averaged (see for more info the LabelFactorizer class in
            scikit_bold.transformers.transformers)

        Returns
        -------
        mvp : Mvp object (see scikit_bold.core module)

        """

        iD = self.identfier
        data_name = op.join(directory, '*', 'mvp_data', '*%s*.hdf5' % iD)
        hdr_name = op.join(directory, '*', 'mvp_data', '*%s*.pickle' % iD)
        data_paths, hdr_paths = glob.glob(data_name), glob.glob(hdr_name)

        n_sub = len(data_paths)
        # Loop over subjects
        for i in range(n_sub):

            # peek at first subject (to get some meta-info)
            if i == 0:
                h5f = h5py.File(data_paths[i], 'r')
                data_tmp = h5f['data'][:]
                h5f.close()
                mvp = cPickle.load(open(hdr_paths[i]))

                if mvp.ref_space == 'epi':
                    msg = 'Cannot concatenate subs from different epi spaces!'
                    raise ValueError(msg)

                # Group labels so we know which conditions (within factorial
                # design) to average
                labfac = LabelFactorizer(grouping)
                mvp.y = labfac.fit_transform(mvp.class_labels)
                mvp.class_labels = list(labfac.get_new_labels())
                mvp.update_metadata()

                data_averaged = np.zeros((mvp.n_class, data_tmp.shape[1]))
                for ii in range(mvp.n_class):
                    class_data = data_tmp[mvp.class_idx[ii], :]
                    data_averaged[ii, :] = np.mean(class_data, axis=0)

                # Pre-allocation
                data = np.zeros((n_sub * mvp.n_class, data_tmp.shape[1]))
                data[(i*mvp.n_class):((i+1)*mvp.n_class), :] = data_averaged

            # This is executed in the rest of the loop
            else:
                tmp = h5py.File(data_paths[i])
                data_tmp = tmp['data'][:]
                tmp.close()
                hdr = cPickle.load(open(hdr_paths[i]))
                labfac = LabelFactorizer(grouping)
                hdr.y = labfac.fit_transform(hdr.class_labels)
                hdr.class_labels = list(labfac.get_new_labels())
                hdr.update_metadata()

                mvp.class_labels.extend(hdr.class_names)

                for ii in range(hdr.n_class):
                    # recycle data_averaged from when i==0
                    class_data = data_tmp[mvp.class_idx[ii], :]
                    data_averaged[ii, :] = np.mean(class_data, axis=0)

                data[(i*mvp.n_class):((i+1)*mvp.n_class), :] = data_averaged

        mvp.X = data
        mvp.sub_name = 'AveragedContrastSubjects'
        self.mvp = mvp

        return mvp

    def write_4D_nifti(self):
        """ Writes a 4D nifti (x, y, z, trials) of an Mvp. """

        print("Creating 4D nifti for %s" % self.mvp.sub_name)
        mvp = self.load()
        img = nib.Nifti1Image(mvp.X, np.eye(4))
        nib.save(img, opj(self.mvp_dir, 'data_4d.nii.gz'))




def sort_numbered_list(stat_list):
    """ Sorts a list containing numbers.

    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts
    the original list accordingly.

    Parameters
    ----------
    stat_list : list[str]
        list with absolute paths to files

    Returns
    -------
    sorted_list : list[str]
        sorted stat_list
    """

    num_list = []
    for path in stat_list:
        num = [str(s) for s in str(op.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return sorted_list
