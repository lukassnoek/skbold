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
from skbold.transformers.transformers import *
from nipype.interfaces import fsl
import nibabel as nib
import warnings


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


class MvpResults(object):
    """ Class that keeps track of model performance over iterations.

    MvpResults keeps track of classification performance across iterations,
    and is able to calculate various performance metrics and additionally
    keeps track of feature importance, which can be operationlized in
    different ways.

    """
    def __init__(self, mvp, iterations, resultsdir='analysis_results',
                 method='voting', verbose=False, feature_scoring=None):

        """ Initializes MvpResults object.

        Parameters
        ----------
        mvp : Mvp object (see scikit_bold.core)
            Needs an Mvp object to extract some meta-data
        iterations : int
            Number of iterations that the analysis is iterated over (i.e.
            number of folds)
        method : str
            Method to calculate performance across iterations. As of now,
            there are two options: 'averaging', which simply calculates
            various performance metrics for each iteration and averages these
            across iterations at the end, and 'voting', which keeps track of
            the predicted class (hard voting) or class probabilities (soft
            voting) across iterations and at the end of all iterations
            determines the class for each sample/trial based on the argmax of
            the mean class probability (soft) or class prediction (hard);
            for more info, see e.g. http://scikit-learn.org/stable/modules/
            generated/sklearn.ensemble.VotingClassifier.html
        verbose : bool
            Whether to print accuracy for each iteration.
        feature_scoring : str (default: None)
            Way to compute importance of features, can be 'accuracy' (default),
            which computes a feature's score as:
            sum(accuracy) / n_selected across iterations. Method 'coefs'
            simply uses the feature weights (linear svm only). Default is
            None (no feature score computation).
        """

        self.method = method
        self.n_iter = iterations
        self.resultsdir = resultsdir
        self.verbose = verbose
        self.sub_name = mvp.sub_name
        self.run_name = mvp.run_name
        self.n_class = mvp.n_class
        self.mask_shape = mvp.mask_shape
        self.mask_index = mvp.mask_index
        self.ref_space = mvp.ref_space
        self.affine = mvp.affine
        self.y_true = mvp.y
        self.X = mvp.X
        self.iter = 0
        self.feature_scoring = feature_scoring
        self.feature_selection = np.zeros(np.sum(mvp.mask_index))
        self.feature_scores = np.zeros(np.sum(mvp.mask_index))
        self.n_features = np.zeros(iterations)
        self.precision = None
        self.accuracy = None
        self.recall = None
        self.conf_mat = np.zeros((mvp.n_class, mvp.n_class))

        if method == 'averaging':
            self.precision = np.zeros(self.n_iter)
            self.accuracy = np.zeros(self.n_iter)
            self.recall = np.zeros(self.n_iter)
        elif method == 'voting':
            self.trials_mat = np.zeros((len(mvp.y), mvp.n_class))

    def update_results(self, test_idx, y_pred, pipeline=None):
        """ Updates results after a cross-validation iteration.

        This method updates the 'counters' for several attributes (relating
        to features and class prediction/accuracy).

        Parameters
        ----------
        test_idx : ndarray[bool] or ndarray[int] (i.e. using fancy indexing)
            Indices for test trials, relativ to mvp.y
        y_pred : ndarray[float]
            Array with predicted classes (if hard-voting/simple predict()) or
            arry with predicted class probabilities (if soft-voting/
            predict_proba()). In the latter case, y_pred.shape =
            [n_class, n_test].
        transformer : transformer object
            This transformer is assumed to have certain attributes, such as
            feature scores (.zvalues) and corresponding indices.
        clf : classifier object (scikit-learn estimator)
            This classifier is assumed to have a _coef attributes, which is
            extracted to keep track of feature weights (and feature importance,
            calculated as weights * feature values).
        """
        fs_method = self.feature_scoring

        if pipeline is not None:

            if hasattr(pipeline, 'best_estimator_'):
                pipeline = pipeline.best_estimator_

            clf = pipeline.named_steps['classifier']
            transformer = pipeline.named_steps['transformer']
            idx, scores = transformer.idx_, transformer.scores_
            self.feature_selection[idx] += 1
            self.n_features[self.iter] = idx.sum()

            if self.feature_scoring == 'distance':
                self.feature_scores[idx] += scores[idx]
            elif fs_method == 'coef':
                coefs = np.mean(np.abs(clf.coef_), axis=0)
                self.feature_scores[idx] += coefs
            elif fs_method == 'accuracy':
                if y_pred.ndim > 1:
                    y_tmp = np.argmax(y_pred, axis=1)
                else:
                    y_tmp = y_pred
                score_tmp = accuracy_score(self.y_true[test_idx], y_tmp)
                self.feature_scores[idx] += score_tmp

        if self.method == 'averaging':
            y_true = self.y_true[test_idx]
            self.conf_mat += confusion_matrix(y_true, y_pred)
            self.precision[self.iter] = precision_score(y_true, y_pred, average='macro')
            self.recall[self.iter] = recall_score(y_true, y_pred, average='macro')
            self.accuracy[self.iter] = accuracy_score(y_true, y_pred)

            if self.verbose:
                print('Accuracy: %f' % self.accuracy[iter])

        elif self.method == 'voting':

            if y_pred.ndim > 1:
                self.trials_mat[test_idx, :] += y_pred
            else:
                self.trials_mat[test_idx, y_pred.astype(int)] += 1

        self.iter += 1

    def compute_score(self, verbose=True):
        """ Computes performance over iterations.

        Computes performance metrics across iterations of the classifier and,
        optionally, computes feature characteristics.

        """
        self.n_features = self.n_features.mean()

        if self.feature_scoring is not None:
            av_feature_info = self.feature_scores / self.feature_selection
            av_feature_info[av_feature_info == np.inf] = 0
            av_feature_info[np.isnan(av_feature_info)] = 0
            self.av_feature_info = av_feature_info

        if self.method == 'voting':

            filter_trials = np.sum(self.trials_mat, axis=1) == 0
            trials_max = np.argmax(self.trials_mat, axis=1) + 1
            trials_max[filter_trials] = 0
            y_pred = trials_max[trials_max > 0]-1
            y_true = self.y_true[trials_max > 0]
            self.conf_mat = confusion_matrix(y_true, y_pred)
            self.precision = precision_score(y_true, y_pred, average='macro')
            self.recall = recall_score(y_true, y_pred, average='macro')
            self.accuracy = accuracy_score(y_true, y_pred)

        elif self.method == 'averaging':

            self.accuracy = self.accuracy.mean()
            self.precision = self.precision.mean()
            self.recall = self.recall.mean()

        if verbose:
            print('Accuracy over iterations: %f' % self.accuracy)

        return self

    def write_results(self, directory, convert2mni=False):
        """ Writes analysis results and feature characteristics to file.

        Parameters
        ----------
        directory : str
            Absolute path to project directory
        resultsdir : str
            name of the results directory   
        convert2mni : bool
            Whether to convert feature scores in epi-space to mni spaces

        """

        results_dir = op.join(directory, self.resultsdir)
        if not op.isdir(results_dir):
            os.makedirs(results_dir)
        filename = op.join(results_dir, '%s_%s_classification.pickle' %
                           (self.sub_name, self.run_name))

        # Remove some attributes to save space
        self.X = None
        self.feature_selection = None

        with open(filename, 'wb') as handle:
            cPickle.dump(self, handle)

        if self.feature_scoring is not None:

            vox_dir = op.join(results_dir, 'vox_results_%s' % self.ref_space)

            if not op.isdir(vox_dir):
                os.makedirs(vox_dir)

            fn = op.join(vox_dir, '%s_%s_%s' % (self.sub_name,
                         self.run_name, self.feature_scoring))

            img = np.zeros(np.prod(self.mask_shape))
            img[self.mask_index] = self.av_feature_info
            img = nib.Nifti1Image(img.reshape(self.mask_shape), self.affine)
            nib.save(img, fn)

            if self.ref_space == 'mni' and convert2mni:
                convert2mni = False

            if self.ref_space == 'epi' and convert2mni:

                mni_dir = op.join(results_dir, 'vox_results_mni')
                if not op.isdir(mni_dir):
                    os.makedirs(mni_dir)

                reg_dir = glob.glob(op.join(directory, self.sub_name, '*',
                                    'reg'))[0]
                ref_file = op.join(reg_dir, 'standard.nii.gz')
                matrix_file = op.join(reg_dir, 'example_func2standard.mat')

                out_file = op.join(mni_dir, op.basename(fn)+'.nii.gz')
                apply_xfm = fsl.ApplyXfm()
                apply_xfm.inputs.in_file = fn + '.nii'
                apply_xfm.inputs.reference = ref_file
                apply_xfm.inputs.in_matrix_file = matrix_file
                apply_xfm.interp = 'trilinear'
                apply_xfm.inputs.out_file = out_file
                apply_xfm.inputs.apply_xfm = True
                apply_xfm.run()

                # Remove annoying .mat files
                to_remove = glob.glob(op.join(os.getcwd(), '*.mat'))
                _ = [os.remove(f) for f in to_remove]

    def compute_and_write(self, directory, convert2mni=False):
        """ Chains compute_score() and write_results(). """
        self.compute_score().write_results(directory, convert2mni)

    def write_results_permutation(self, directory, perm_number):
        """ Writes permutation results.

        Instead of the 'regular' write_results() method, this method stores
        only the confusion matrix and voxel scores, to avoid saving huge
        amounts of data on disk - due to inefficient pickle storage - as
        every permutation result may amount to >1 GB of data.

        Parameters
        ----------
        directory : str
            Absolute path to project directory

        perm_number : str (or int/float)
            Number of the permutation
        """

        if type(perm_number) == int or type(perm_number) == float:
            perm_number = str(perm_number)

        perm_dir = op.join(directory, 'permutation_results')
        if not op.isdir(perm_dir):
            os.makedirs(perm_dir)

        current_perm_dir = op.join(perm_dir, 'perm_%s' % perm_number)
        if not op.isdir(current_perm_dir):
            os.makedirs(current_perm_dir)

        filename = op.join(current_perm_dir, '%s_%s_confmat.npy' %
                           (self.sub_name, self.run_name))

        np.save(filename, self.conf_mat)

        if self.feature_scoring is not None:

            vox_dir = op.join(current_perm_dir, 'vox_results_%s' % self.ref_space)

            if not op.isdir(vox_dir):
                os.makedirs(vox_dir)

            fn = op.join(vox_dir, '%s_%s_%s' % (self.sub_name,
                         self.run_name, self.feature_scoring))

            img = np.zeros(np.prod(self.mask_shape))
            img[self.mask_index] = self.av_feature_info
            img = nib.Nifti1Image(img.reshape(self.mask_shape), self.affine)
            nib.save(img, fn)


class MvpAverageResults(object):
    """ Class that averages individual subject classification performance.

    Is able to load individual result-files, process/analyze them and write
    their average/processed results as a pandas dataframe.

    """

    def __init__(self, directory, resultsdir='analysis_results', params=None,
                 threshold=None, cleanup=True):
        """ Initializes MvpAverageResults object.

        Parameters
        ----------
        directory : str
            Absolute path to where individual classification files are located.
        threshold : int
            Threshold for summation-accuracy metric of voxel scores
            (i.e. score = sum(voxel_scores > threshold)).
        cleanup = bool
            Whether to clean up subject-specific nifti-files
        params : dict
            Dictionary with analysis parameters
        """

        self.directory = directory
        self.threshold = threshold
        self.params = params
        self.resultsdir = resultsdir
        self.cleanup = cleanup
        self.threshold = threshold
        self.cleanup = cleanup
        self.df_ = pd.DataFrame()

    def average(self):
        """ Loads and computes average performance metrics. """

        results_dir = op.join(self.directory, self.resultsdir)
        files = glob.glob(op.join(results_dir, '*.pickle'))

        for i, f in enumerate(files):
            results = cPickle.load(open(f, 'rb'))
            tmp = {'Sub_name': results.sub_name,
                   'Accuracy': results.accuracy,
                   'Precision': results.precision,
                   'Recall': results.recall,
                   'n_feat': results.n_features}
            self.df_ = self.df_.append(pd.DataFrame(tmp, index=[i]))

        if not self.threshold:
            n_class = results.n_class
            self.threshold = (1/n_class) + .2 * (1/n_class)

        tmp = {'Sub_name': 'Average',
               'Accuracy': self.df_['Accuracy'].mean(),
               'Precision': self.df_['Precision'].mean(),
               'Recall': self.df_['Recall'].mean(),
               'n_feat': self.df_['n_feat'].mean()}

        self.df_ = self.df_.append(pd.DataFrame(tmp, index=[i+1]))
        cols = list(self.df_)
        cols.insert(0, cols.pop(cols.index('Sub_name')))
        self.df_ = self.df_.ix[:, cols]

        if self.cleanup:
            _ = [os.remove(f) for f in files]

        print(self.df_)

        filename = op.join(results_dir, 'average_results.csv')
        self.df_.to_csv(filename, sep='\t', header=True, index=False)

        if self.params:
            file2open = op.join(self.directory, 'analysis_parameters.json')
            with open(file2open, 'w') as f:
                json.dump(self.params, f)

        feature_files = glob.glob(op.join(results_dir, 'vox_results_mni',
                                          '*.nii*'))
        if len(feature_files) > 0:

            shape = (91, 109, 91)  # hard-coded mni-shape
            s = np.zeros((len(feature_files), shape[0], shape[1], shape[2]))

            for i, feature_file in enumerate(feature_files):

                data = nib.load(feature_file).get_data()
                s[i, :, :, :] = data

            metric = feature_file.split('_')[-1].split('.')[0]

            if metric == 'accuracy':
                s = (s > self.threshold).astype(int).sum(axis=0)
            else:
                s = s.mean(axis=0)

            if self.cleanup:
                cmd = 'rm %s/*%s*.nii' % results_dir
                _ = os.system(cmd)

            fn = op.join(results_dir, 'AverageScores')
            img = nib.Nifti1Image(s, np.eye(4))
            nib.save(img, fn)


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
