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
from scikit_bold.transformers.transformers import *
from nipype.interfaces import fsl
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
        iD = self.identfier
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
        num = [str(s) for s in str(os.path.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return sorted_list


class MvpResults(object):
    """ Contains info about model performance across iterations
    """
    def __init__(self, mvp, iterations, method='voting',
                 verbose=0, condition_name=None):

        self.method = method
        self.n_iter = iterations
        self.verbose = verbose
        self.sub_name = mvp.sub_name
        self.run_name = mvp.run_name
        self.n_class = mvp.n_class
        self.condition_name = condition_name
        self.mask_shape = mvp.mask_shape
        self.mask_index = mvp.mask_index
        self.ref_space = mvp.ref_space
        self.affine = mvp.affine
        self.y_true = mvp.y
        self.iter = 0

        self.feature_selection = np.zeros(np.sum(mvp.mask_index))
        self.feature_zscores = np.zeros(np.sum(mvp.mask_index))
        self.feature_scores = np.zeros(np.sum(mvp.mask_index))
        self.feature_weights = np.zeros(np.sum(mvp.mask_index))
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

    def update_results(self, test_idx, y_pred, feature_idx=None, feature_zscores=None, feature_weights=None):
        """ Updates results after a cross-validation iteration """
        
        compute_features = not ((feature_idx is None) or (feature_zscores is None) or (feature_weights is None))
        self.compute_features = compute_features

        if compute_features:
            self.feature_zscores[feature_idx] += feature_zscores[feature_idx]     
            self.feature_weights[feature_idx] += np.mean(np.abs(feature_weights))
            self.feature_selection += feature_idx.astype(int) 
            self.n_features[self.iter] = feature_idx.sum()
            
            if y_pred.ndim > 1:
                y_tmp = np.argmax(y_pred, axis=1)
            else:
                y_tmp = y_pred

            self.feature_scores[feature_idx] += accuracy_score(self.y_true[test_idx], y_tmp)

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

        return self

    def compute_score(self):

        self.n_features = self.n_features.mean()
        if self.compute_features:
            self.feature_zscores = self.feature_zscores / self.feature_selection
            self.feature_scores = self.feature_scores / self.feature_selection
            self.feature_weights = self.feature_weights / self.feature_selection
            self.feature_zscores[np.isnan(self.feature_zscores)] = 0
            self.feature_scores[np.isnan(self.feature_scores)] = 0
            self.feature_weights[np.isnan(self.feature_weights)] = 0

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

        print('Accuracy over iterations: %f' % self.accuracy)
        return self

    def write_results(self, directory, convert2mni=False):

        filename = os.path.join(directory, '%s_%s_classification.pickle' % \
                                (self.sub_name, self.run_name))
        with open(filename, 'wb') as handle:
            cPickle.dump(self, handle)

        if self.compute_features:
            vox_dir = os.path.join(directory, 'vox_results_%s' % self.ref_space)
            if not os.path.isdir(vox_dir):
                os.makedirs(vox_dir)

            # Write feature-selection scores (zvalues)
            fn_zscores = os.path.join(vox_dir, '%s_%s_FeatureZscores' % \
                                    (self.sub_name, self.run_name))

            img = np.zeros(np.prod(self.mask_shape))
            img[self.mask_index] = self.feature_zscores
            img = nib.Nifti1Image(img.reshape(self.mask_shape), self.affine)
            nib.save(img, fn_zscores)

            # Write feature weights
            fn_weights = os.path.join(vox_dir, '%s_%s_FeatureWeights' % \
                                    (self.sub_name, self.run_name))
            img = np.zeros(np.prod(self.mask_shape))
            img[self.mask_index] = self.feature_weights
            img = nib.Nifti1Image(img.reshape(self.mask_shape), self.affine)
            nib.save(img, fn_weights)

            # Write feature scores
            fn_scores = os.path.join(vox_dir, '%s_%s_FeatureScores' % \
                                    (self.sub_name, self.run_name))
            img = np.zeros(np.prod(self.mask_shape))
            img[self.mask_index] = self.feature_scores
            img = nib.Nifti1Image(img.reshape(self.mask_shape), self.affine)
            nib.save(img, fn_scores)

            if self.ref_space == 'epi' and convert2mni:

                files2transform = [fn_zscores +'.nii', fn_weights+'.nii', fn_scores+'.nii']
                mni_dir = os.path.join(directory, 'vox_results_mni')
                if not os.path.isdir(mni_dir):
                    os.makedirs(mni_dir)

                reg_dir = glob.glob(opj(directory, '*', self.sub_name, '*', 'reg'))[0]
                ref_file = opj(reg_dir, 'standard.nii.gz')
                matrix_file = opj(reg_dir, 'example_func2standard.mat')
            
                for f in files2transform:
                    out_file = opj(mni_dir, os.path.basename(f)+'.gz')
                    apply_xfm = fsl.ApplyXfm()
                    apply_xfm.inputs.in_file = f
                    apply_xfm.inputs.reference = ref_file
                    apply_xfm.inputs.in_matrix_file = matrix_file
                    apply_xfm.interp = 'trilinear'
                    apply_xfm.inputs.out_file = out_file
                    apply_xfm.inputs.apply_xfm = True
                    apply_xfm.run()

                # Remove annoying .mat files
                _ = [os.remove(f) for f in glob.glob(opj(os.getcwd(), '*.mat'))]


class MvpAverageResults(object):
    """
    Object that is able to load individual result-files, process/analyze
    them and write their average/processed results as a pandas dataframe
    """

    def __init__(self, directory, identifier, compute_features=False, params=None, threshold=None, cleanup=True):
        self.directory = directory
        self.threshold = threshold
        self.identifier = identifier
        self.compute_features = compute_features
        self.params = params
        self.cleanup = cleanup
        self.threshold = threshold
        self.df_ = pd.DataFrame()

    def load(self):

        files = glob.glob(os.path.join(self.directory, '*%s*.pickle' % \
                                       self.identifier))
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

        _ = [os.remove(f) for f in files]

        print(self.df_)

        if self.compute_features:
            vox_files_dir = os.path.join(self.directory, 'vox_results_mni')
            zscore_files = glob.glob(os.path.join(vox_files_dir, '*%s*FeatureZscores*.nii.gz' % \
                                     self.identifier))
            
            weight_files = glob.glob(os.path.join(vox_files_dir, '*%s*FeatureWeights*.nii.gz' % \
                                     self.identifier))
        
            score_files = glob.glob(os.path.join(vox_files_dir, '*%s*FeatureScores*.nii.gz' % \
                                     self.identifier))

            mask_shape = (91, 109, 91) # hard-coded mni-shape
            z = np.zeros((len(zscore_files), mask_shape[0], mask_shape[1], mask_shape[2]))
            w = np.zeros((len(weight_files), mask_shape[0], mask_shape[1], mask_shape[2]))
            s = np.zeros((len(score_files), mask_shape[0], mask_shape[1], mask_shape[2]))

            for i, (zscore, weight, score) in enumerate(zip(zscore_files, weight_files, score_files)):
                z[i, :, :, :] = nib.load(zscore).get_data()
                w[i, :, :, :] = nib.load(weight).get_data()
                s[i, :, :, :] = (nib.load(score).get_data() > self.threshold).astype(int)
                
            if self.cleanup:
                _ = os.system('rm %s/*%s*.nii' % (self.directory, self.identifier))

            filename = os.path.join(self.directory, '%s_AverageZscores' % self.identifier)
            img = nib.Nifti1Image(z.mean(axis=0), np.eye(4))
            nib.save(img, filename)

            filename = os.path.join(self.directory, '%s_AverageWeights' % self.identifier)
            img = nib.Nifti1Image(w.mean(axis=0), np.eye(4))
            nib.save(img, filename)

            filename = os.path.join(self.directory, '%s_AverageScores' % self.identifier)
            img = nib.Nifti1Image(s.sum(axis=0), np.eye(4))
            nib.save(img, filename)

        return self

    def write(self):
        filename = os.path.join(self.directory, 'average_results_%s.csv' % \
                                self.identifier)
        self.df_.to_csv(filename, sep='\t', header=True, index=False)

        if self.params:
            with open(os.path.join(self.directory, 'analysis_parameters.json'), 'w') as f:
                json.dump(self.params, f)


class AnalysisPermuter():

    def __init__(self, project_dir):
        pass

