# Class to keep track of cross-validation iterations of a classification
# pipeline.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
import cPickle
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
     confusion_matrix
from skbold.transformers import *
import numpy as np
import os.path as op
import os
import glob
from nipype.interfaces import fsl
from itertools import chain, combinations
from scipy.misc import comb
import nibabel as nib


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

        n_voxels = np.sum(mvp.mask_index)

        if 'coef' in feature_scoring:
            n_models = mvp.n_class * (mvp.n_class - 1) / 2
            self.feature_scores = np.zeros((n_voxels, n_models))
        elif feature_scoring == 'accuracy':
            self.feature_scores = np.zeros((n_voxels, mvp.n_class))
        else: # if 'distance'
            self.feature_scores = np.zeros(n_voxels)

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

        # If pipeline is None, assume that we don't keep track of voxels.
        if pipeline is not None:

            # If GridSearchCv, extract best estimator as pipeline.
            if hasattr(pipeline, 'best_estimator_'):
                pipeline = pipeline.best_estimator_

            clf = pipeline.named_steps['classifier']
            transformer = pipeline.named_steps['transformer']
            idx, scores = transformer.idx_, transformer.scores_
            self.feature_selection[idx] += 1
            self.n_features[self.iter] = idx.sum()

            if self.feature_scoring == 'distance':
                self.feature_scores[idx] += scores[idx]
            elif 'coef' in fs_method:
                self.feature_scores[idx] += clf.coef_.T
            elif fs_method == 'accuracy':
                if y_pred.ndim > 1:
                    # if output is proba estimates, then do soft vote
                    y_tmp = np.argmax(y_pred, axis=1)
                else:
                    y_tmp = y_pred

                cm_tmp = confusion_matrix(self.y_true[test_idx], y_tmp)
                score_tmp = np.diag(cm_tmp / cm_tmp.sum(axis=1))
                score_tmp = np.expand_dims(score_tmp, 0)
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
        fs = self.feature_scoring

        if fs is not None:

            if fs == 'coefovr':
                """
                Here, we add the coefficients from the pairwise classifications
                together, e.g. for class 'A', we add 'A vs. B' and 'A vs. C'
                together, and if class 'A' is the negative class, we multiply
                the coefficient by -1. Note that sklearn's multiclass models
                reverse the labels for some reason (e.g. in the class 0 vs 1
                classification, 0 becomes the positive label, and 1 becomes the
                negative label), that's why the coefs are all reversed at the
                beginning.
                """
                av_feature_info = self.feature_scores * -1
                n, k = self.n_class, 2
                count = comb(n, k, exact=True)
                index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                                    int, count=count*k)
                c = index.reshape(-1, k)
                store = np.zeros((av_feature_info.shape[0], av_feature_info.shape[1], n, 2))

                for clas in range(n):

                    present = 0
                    for i in range(n):
                        if any(c[i, :] == clas):
                            if (c[i, :] == clas)[0]:
                                store[:, :, clas, present] = av_feature_info[:, :, i] * -1
                            else:
                                store[:, :, clas, present] = av_feature_info[:, :, i]

                            present += 1

                store[store < 0] = 0
                self.feature_scores = store.sum(axis=3)

            elif fs == 'coefovo':
                av_feature_info = self.feature_scores * -1

            # n_selected = (av_feature_info != 0.0).sum(axis=0)
            # av_feature_info = self.feature_scores.sum(axis=0) / n_selected
            if 'coef' in fs or fs == 'accuracy':
                self.feature_selection = np.expand_dims(self.feature_selection, axis=1)

            av_feature_info = self.feature_scores / self.feature_selection
            av_feature_info[av_feature_info == np.inf] = 0
            av_feature_info[np.isnan(av_feature_info)] = 0

            """
            Here, the std should be calculated, but I have now clue
            how to do this properly...
            """
            # std_av_features = (av_feature_info-self.feature_scores * -1)
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

        if self.feature_scoring is not None:

            vox_dir = op.join(results_dir, 'vox_results_%s' % self.ref_space)

            if not op.isdir(vox_dir):
                os.makedirs(vox_dir)

            fn = op.join(vox_dir, '%s_%s_%s' % (self.sub_name,
                         self.run_name, self.feature_scoring))

            if self.feature_scoring == 'accuracy' or 'coef' in self.feature_scoring:
                img = np.zeros((np.prod(self.mask_shape), self.n_class))
                img[self.mask_index, :] = self.av_feature_info
                xs, ys, zs = self.mask_shape
                img = img.reshape((xs, ys, zs, self.n_class))
            else:
                img = np.zeros(np.prod(self.mask_shape))
                img[self.mask_index] = self.av_feature_info
                img = img.reshape(self.mask_shape)

            img = nib.Nifti1Image(img, self.affine)
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

        # Remove some attributes to save space
        self.X = None
        self.feature_selection = None
        self.feature_scores = None
        self.mask_index = None

        with open(filename, 'wb') as handle:
            cPickle.dump(self, handle)

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