# Class to implement a stacking classifier that trains a meta-classifier
# on classifiers trained on different feature sets from different ROIs.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

# Note: this implementation was inspired by the code of S. Rashka
# (http://sebastianraschka.com/Articles/2014_ensemble_classifier.html)

import glob
import numpy as np
import os
import os.path as op
import joblib
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from skbold.transformers import RoiIndexer, MeanEuclidean, \
    IncrementalFeatureCombiner
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from copy import copy, deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.grid_search import GridSearchCV
import warnings
from skbold.core import convert2epi

warnings.filterwarnings('ignore')  # hack to turn off UndefinedMetricWarning
import skbold

roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs', 'harvard_oxford')


class RoiStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    This scikit-learn-style classifier implements a stacking classifier
    that fits a base-classifier on multiple brain-regions separately and
    subsequently trains a meta-classifier on the outputs of the base-
    classifiers on the separate brain-regions.


    Parameters
    ----------
    mvp : mvp-object
        An custom object from the skbold package containing data (X, y)
        and corresponding meta-data (e.g. mask info)
    preproc_pipe : object
        A scikit-learn Pipeline object with desired preprocessing steps
        (e.g. scaling, additional feature selection). Defaults to only
        scaling and univariate-feature-selection by means of highest
        mean-euclidean differences (see
        ``skbold.transformers.mean_euclidean``).
    base_clf : object
        A scikit-learn style classifier (implementing fit(), predict(),
        and predict_proba()), that is able to be used in Pipelines.
    meta_clf : object
        A scikit-learn style classifier.
    mask_type : str
        Can be 'unilateral' or 'bilateral', which will use all masks from
        the corresponding Harvard-Oxford Cortical (lateralized) atlas.
        Alternatively, it may be an absolute path to a directory containing
        a custom set of masks as nifti-files (default: 'unilateral').
    meta_gs : list or ndarray
        Optional parameter-grid over which to perform gridsearch.
    n_cores : int
        Number of CPU-cores on which to perform the fitting procedure
        (here, outer-folds are parallelized).

    Attributes
    ----------
    train_scores : ndarray
        Accuracy-scores per brain region (averaged over outer-folds) on the
        training (fit) phase.
    test_scores : ndarray
        Accuracy-scores per brain region (averaged over outer- and inner-folds)
        on the test phase.
    masks : list of str
        List of absolute paths to found masks.
    stck_train : ndarray
        Array with outputs from base-classifiers fit on train-set.
    stck_test : ndarray
        Array with outputs from base-classifiers generalized to test-set.
    """

    def __init__(self, mvp, preproc_pipe='default', base_clf=None,
                 meta_clf=None,
                 mask_type='unilateral', proba=True, folds=10, meta_fs='univar',
                 meta_gs=None, n_cores=1):

        self.mvp = copy(mvp)
        self.mvp.X = None
        self.n_class = mvp.n_class
        self.mask_type = mask_type
        self.proba = proba
        self.n_cores = n_cores

        if preproc_pipe == 'default':
            scaler = StandardScaler()
            transformer = MeanEuclidean(cutoff=1, normalize=False)
            preproc_pipe = Pipeline([('transformer', transformer),
                                     ('scaler', scaler)])

        if base_clf is None:
            base_clf = SVC(C=1.0, kernel='linear', probability=True,
                           decision_function_shape='ovo')
        self.base_clf = base_clf

        base_pipe = preproc_pipe.steps
        base_pipe.extend([('base_clf', self.base_clf)])
        self.base_pipe = Pipeline(base_pipe)
        self.base_pipes = []

        if meta_clf is None:
            meta_clf = LogisticRegression(multi_class='multinomial',
                                          C=0.1, solver='lbfgs')
        self.meta_fs = meta_fs
        meta_pipe = Pipeline([('selector', meta_fs),
                              ('scaler', scaler),
                              ('meta_clf', meta_clf)])

        if meta_gs is not None:
            params = dict(selector__cutoff=meta_gs)
            meta_pipe = GridSearchCV(meta_pipe, params, error_score=0)

        self.meta_pipe = meta_pipe

        # Glob masks
        if mask_type not in ['unilateral', 'bilateral']:
            # It is assumed that a directory with masks is inputted
            self.masks = glob.glob(op.join(mask_type, '*nii.gz'))
        else:
            mask_dir = op.join(roi_dir, mask_type)
            self.masks = glob.glob(op.join(mask_dir, '*nii.gz'))

        if not self.masks:
            raise ValueError('No masks found in specified directory!')

        if mvp.ref_space == 'epi':

            for i, mask in enumerate(self.masks):

                if op.basename(mask)[0:2] in ['L_', 'R_']:
                    laterality = 'unilateral'
                else:
                    laterality = 'bilateral'

                main_dir = op.dirname(mvp.directory)
                epi_dir = op.join(main_dir, 'epi_masks', laterality)

                if not op.isdir(epi_dir):
                    os.makedirs(epi_dir)

                epi_name = op.basename(mask)[:-7]
                epi_exists = glob.glob(
                    op.join(epi_dir, '*%s*.nii.gz' % epi_name))

                if epi_exists:
                    self.masks[i] = epi_exists[0]
                else:
                    reg_dir = op.join(mvp.directory, 'reg')
                    self.masks[i] = convert2epi(mask, reg_dir, epi_dir)[0]

        self.folds = folds

        # Metrics
        self.train_roi_scores = None
        self.test_roi_scores = None
        self.stack_dir = op.join(op.dirname(mvp.directory), 'stack_dir')

        if not op.isdir(self.stack_dir):
            os.makedirs(self.stack_dir)
        else:
            _ = [os.remove(f) for f in
                 glob.glob(op.join(self.stack_dir, '*.npy'))]

    def _fit_base(self, X, y, n_cores=1):

        nr_folds = range(self.folds)

        Parallel(n_jobs=n_cores, verbose=5)(delayed(
            _fit_base_parallel)(self, X, y, i) for i in nr_folds)

        stacks = sorted(glob.glob(op.join(self.stack_dir, 'features*.npy')))
        stck_train = np.stack([np.load(s) for s in stacks]).mean(axis=0)
        self.stck_train = stck_train.reshape(stck_train.shape[0], -1)
        scores = sorted(glob.glob(op.join(self.stack_dir, 'scores*.npy')))
        self.train_roi_scores = np.stack([np.load(s) for s in scores]).mean(
            axis=0).mean(axis=0)
        self.base_pipes = sorted(glob.glob(op.join(self.stack_dir, 'pipes*')))

    def _fit_meta(self, y):

        if self.meta_fs.__class__ == IncrementalFeatureCombiner:
            if self._is_gridsearch(self.meta_pipe):
                self.meta_pipe.estimator.set_params(
                    selector__scores=self.train_roi_scores)
            else:
                self.meta_pipe.set_params(
                    selector__scores=self.train_roi_scores)

        self.meta_pipe.fit(self.stck_train, y)

        if self._is_gridsearch(self.meta_pipe):
            self.best_roi_idx = self.meta_pipe.best_estimator_.named_steps[
                'selector'].idx_
        else:
            self.best_roi_idx = self.meta_pipe.named_steps['selector'].idx_

    def _predict_base(self, X, y=None):

        n_trials = X.shape[0]
        n_class = self.n_class
        n_masks = len(self.masks)
        n_inner = self.n_inner_cv
        n_outer = self.folds

        if self.proba:
            stck_fts = np.zeros((n_trials, n_class, n_masks, n_outer, n_inner))
        else:
            stck_fts = np.zeros((n_trials, n_masks, n_outer, n_inner))

        scores = np.zeros((n_outer, n_inner, n_masks, n_class))

        for i in range(len(self.base_pipes)):

            if all(isinstance(p, str) for p in self.base_pipes):
                outer_pipe = joblib.load(self.base_pipes[i])
                os.remove(self.base_pipes[i])
            else:
                outer_pipe = self.base_pipes[i]

            for ii, inner_pipe in enumerate(outer_pipe):

                for iii, mask_pipe in enumerate(inner_pipe):

                    pred = mask_pipe.predict(X)
                    scores[i, ii, iii, :] = precision_score(y, pred,
                                                            average=None)

                    if self.proba:
                        stck_fts[:, :, iii, i, ii] = mask_pipe.predict_proba(X)
                    else:
                        stck_fts[:, iii, i, ii] = pred

        self.stck_test = stck_fts.mean(axis=-1).mean(axis=-1)
        self.test_roi_scores = scores.mean(axis=0).mean(axis=0)

    def _predict_meta(self, X, y=None):

        if self.proba:
            shape = self.stck_test.shape
            self.stck_test = self.stck_test.reshape(
                (shape[0], np.prod(shape[1:])))
            meta_pred = self.meta_pipe.predict_proba(self.stck_test)
        else:
            meta_pred = self.meta_pipe.predict(self.stck_test)

        return meta_pred

    def fit(self, X, y):
        """ Fits RoiStackingClassfier.

        Parameters
        ----------
        X : ndarray
            Array of shape = [n_samples, n_features].
        y : list or ndarray of int or float
            List or ndarray with floats/ints corresponding to labels.

        Returns
        -------
        self : object
            RoiStackingClassifier instance with fitted parameters.
        """

        self.n_inner_cv = (y == 0).sum()  # assumes balanced classes
        self._fit_base(X, y, n_cores=self.n_cores)
        self._fit_meta(y)

        return self

    def predict(self, X, y=None):
        """ Predict class given  RoiStackingClassifier.

        Parameters
        ----------
        X : ndarray
            Array of shape = [n_samples, n_features].

        Returns
        -------
        meta_pred : ndarray
            Array with class predictions.
        """

        self._predict_base(X, y=y)
        meta_pred = self._predict_meta(X, y=y)

        return meta_pred

    def score(self, X, y):
        """ Scoring function calculating accuracy given predictions.

        X : ndarray
            Array of shape = [n_samples, n_features]
        y : list or ndarray of int or float
            List or ndarray with floats/ints corresponding to labels.

        Returns
        -------
        score : float
            Accuracy of predictions on the test-set.
        """
        pred = self.predict(X, y)
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)

        score = np.mean(pred == y)
        print('Score: %f' % score)
        return score

    def _is_gridsearch(self, pipe):

        if hasattr(pipe, 'estimator'):
            return True
        else:
            return False


def _fit_base_parallel(rsc, X, y, i):
    """ Parallelized fitting function (cannot be method due to the fact
    that joblib is unable to pickle class methods. """

    n_masks = len(rsc.masks)
    n_outer_cv = rsc.folds
    n_masks = len(rsc.masks)
    n_class = rsc.mvp.n_class
    n_trials = X.shape[0]
    n_inner_cv = (y == 0).sum()

    folds = StratifiedKFold(y, n_folds=n_inner_cv, shuffle=True,
                            random_state=(i + 1))

    if rsc.proba:  # trials X masks X nr. classes (proba output)
        stck_fts = np.zeros((n_trials, n_class, n_masks))
    else:  # discrete class output
        stck_fts = np.zeros((n_trials, n_masks))

    scores = np.zeros((n_inner_cv, n_masks, n_class))

    inner_pipes = []
    for ii, fold in enumerate(folds):
        train_idx, test_idx = fold
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]

        mask_pipes = []
        for iii, mask in enumerate(rsc.masks):
            base_pipe = deepcopy(rsc.base_pipe)
            ri = RoiIndexer(rsc.mvp, mask, mask_threshold=0)
            base_tmp = base_pipe.steps
            base_tmp.insert(0, ('roiindexer', ri))
            base_pipe = Pipeline(base_tmp)
            base_pipe.fit(X_train, y_train)
            pred = base_pipe.predict(X_test)

            if rsc.proba:
                stck_fts[test_idx, :, iii] = base_pipe.predict_proba(X_test)
            else:
                stck_fts[test_idx, iii] = pred

            scores[ii, iii, :] = precision_score(y_test, pred, average=None)
            mask_pipes.append(base_pipe)

        inner_pipes.append(mask_pipes)

    fn = op.join(rsc.stack_dir, 'features_fold_%i.npy' % (i + 1))
    np.save(fn, stck_fts)
    fn = op.join(rsc.stack_dir, 'scores_fold_%i.npy' % (i + 1))
    np.save(fn, scores)
    fn = op.join(rsc.stack_dir, 'pipes_fold_%i.pickle' % (i + 1))
    joblib.dump(inner_pipes, fn, compress=3)
