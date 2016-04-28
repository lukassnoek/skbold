# Class to implement a stacking classifier that trains a meta-classifier
# on classifiers trained on different feature sets from different ROIs.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

# Note: this implementation was inspired by the code of S. Rashka
# (http://sebastianraschka.com/Articles/2014_ensemble_classifier.html)

import glob
import numpy as np
import os.path as op
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from skbold.transformers import RoiIndexer, MeanEuclidean, IncrementalFeatureCombiner
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from copy import copy, deepcopy
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings('ignore')  # hack to turn off UndefinedMetricWarning

import skbold
roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs', 'harvard_oxford')


class RoiStackingClassifier(BaseEstimator, ClassifierMixin):
    """ Stacking classifier for an ensemble of patterns from different ROIs.
    """

    def __init__(self, mvp, preproc_pipe='default', base_clf=None, meta_clf=None,
                 mask_type='unilateral', proba=True, folds=10, meta_fs='univar',
                 meta_gs=None, n_cores=1):
        """ Initializes RoiStackingClassifier object.

        Parameters
        ----------
        mvp : mvp-object
            An custom object from the skbold package containing data (X, y)
            and corresponding meta-data (e.g. mask info)
        preproc_pipeline : object
            A scikit-learn Pipeline object with desired preprocessing steps
            (e.g. scaling, additional feature selection)
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
        """

        self.mvp = mvp
        self.n_class = mvp.n_class
        self.mask_type = mask_type
        self.proba = proba
        self.n_cores = n_cores

        # If no preprocessing pipeline is defined, we'll assume that at least
        # scaling and minor (univariate) feature selection is desired.
        if preproc_pipe == 'default':
            scaler = StandardScaler()
            transformer = MeanEuclidean(cutoff=1, normalize=False)
            preproc_pipe = Pipeline([('transformer', transformer),
                                     ('scaler', scaler)])

        """ STEP 2a: BASE CLASSIFIER """
        if base_clf is None:
            base_clf = SVC(C=1.0, kernel='linear', probability=True,
                           decision_function_shape='ovo')
        self.base_clf = base_clf

        """ STEP 2b: PREPROC+BASE """
        base_pipe = preproc_pipe.steps
        base_pipe.extend([('base_clf', self.base_clf)])
        self.base_pipe = Pipeline(base_pipe)
        self.base_pipes = []

        """ STEP 3a: META FEATURE SELECTION + CLASSIFICATION """
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

        self.folds = folds

        # Metrics
        self.train_roi_scores = None
        self.test_roi_scores = None

    def fit(self, X, y):
        """ Fits RoiStackingTransformer.

        Parameters
        ----------
        X : ndarray
            Array of shape = [n_samples, n_features]. If None (default), X
            is drawn from the mvp object.
        y : List[str] or numpy ndarray[str]
            List or ndarray with floats corresponding to labels.
        """

        n_masks = len(self.masks)
        n_outer_cv = self.folds
        n_masks = len(self.masks)
        n_class = self.mvp.n_class
        n_trials = X.shape[0]
        n_inner_cv = (y == 0).sum()

        if self.proba:  # trials X masks X nr. classes (proba output)
            stck_fts = np.zeros((n_trials, n_class, n_masks, n_outer_cv))
        else:  # discrete class output
            stck_fts = np.zeros((n_trials, n_masks, n_outer_cv))

        scores = np.zeros((n_outer_cv, n_inner_cv, n_masks, n_class))
        outer_pipes = []
        for i in range(n_outer_cv):
            folds = StratifiedKFold(y, n_folds=n_inner_cv, shuffle=True,
                                    random_state=(i + 1))

            inner_pipes = []
            for ii, fold in enumerate(folds):
                train_idx, test_idx = fold
                X_train, y_train = X[train_idx, :], y[train_idx]
                X_test, y_test = X[test_idx, :], y[test_idx]

                mask_pipes = []
                for iii, mask in enumerate(self.masks):
                    base_pipe = deepcopy(self.base_pipe)
                    ri = RoiIndexer(self.mvp, mask, mask_threshold=0)
                    base_tmp = base_pipe.steps
                    base_tmp.insert(0, ('roiindexer', ri))
                    base_pipe = Pipeline(base_tmp)
                    base_pipe.fit(X_train, y_train)
                    pred = base_pipe.predict(X_test)

                    if self.proba:
                        stck_fts[test_idx, :, iii, i] = base_pipe.predict_proba(X_test)
                    else:
                        stck_fts[test_idx, iii, i] = pred

                    scores[i, ii, iii, :] = precision_score(y_test, pred, average=None)
                    mask_pipes.append(base_pipe)

                inner_pipes.append(mask_pipes)
            outer_pipes.append(inner_pipes)
        self.base_pipes = outer_pipes

        mean_scores = scores.mean(axis=0).mean(axis=0)
        stck_fts = stck_fts.mean(axis=-1)
        shape = stck_fts.shape
        stck_tmp = stck_fts.reshape((shape[0], shape[1] * shape[2]))

        if self.meta_fs.__class__ == IncrementalFeatureCombiner:
            if self.is_gridsearch(self.meta_pipe):
                self.meta_pipe.estimator.set_params(
                    selector__scores=mean_scores)
            else:
                self.meta_pipe.set_params(selector__scores=mean_scores)

        self.meta_pipe.fit(stck_tmp, y)

        # Save some stuff for exploratory stuff / debugging
        self.stck_train = stck_tmp
        self.train_roi_scores = mean_scores

        if self.is_gridsearch(self.meta_pipe):
            self.best_roi_idx = self.meta_pipe.best_estimator_.named_steps[
                'selector'].idx_
        else:
            self.best_roi_idx = self.meta_pipe.named_steps['selector'].idx_

        return self

    def predict(self, X, y=None):
        """ Predict class given  RoiStackingTransformer.

        Parameters
        ----------
        X : ndarray
            Array of shape = [n_samples, n_features].
        """

        n_trials = X.shape[0]
        n_class = self.n_class

        n_masks = len(self.masks)
        n_inner = len(self.base_pipes[0])
        n_outer = len(self.base_pipes)

        if self.proba:
            stck_fts = np.zeros((n_trials, n_class, n_masks, n_outer, n_inner))
        else:
            stck_fts = np.zeros((n_trials, n_masks, n_outer, n_inner))

        scores = np.zeros((n_outer, n_inner, n_masks, n_class))

        for i, outer_pipe in enumerate(self.base_pipes):

            for ii, inner_pipe in enumerate(outer_pipe):

                for iii, mask_pipe in enumerate(inner_pipe):

                    pred = mask_pipe.predict(X)
                    scores[i, ii, iii, :] = precision_score(y, pred, average=None)

                    if self.proba:
                        stck_fts[:, :, iii, i, ii] = mask_pipe.predict_proba(X)
                    else:
                        stck_fts[:, iii, i, ii] = pred

        stck_fts = stck_fts.mean(axis=-1).mean(axis=-1)

        if self.proba:
            shape = stck_fts.shape
            stck_fts = stck_fts.reshape((shape[0], np.prod(shape[1:])))
            meta_pred = self.meta_pipe.predict_proba(stck_fts)
        else:
            meta_pred = self.meta_pipe.predict(stck_fts)

        self.stck_test = stck_fts
        self.test_roi_scores = scores.mean(axis=0).mean(axis=0)

        return meta_pred

    def is_gridsearch(self, pipe):

        if hasattr(pipe, 'estimator'):
            return True
        else:
            return False


if __name__ == '__main__':

    from skbold.utils import DataHandler, MvpResults, MvpAverageResults
    from sklearn.cross_validation import cross_val_predict

    sub_dirs = glob.glob('/media/lukas/data/DecodingEmotions/Validation_set/glm_zinnen/sub*')

    # Params
    mask_dir = 'unilateral'#'/home/lukas/test_rois'
    test_folds = 20
    cv_folds = 10
    meta_fs = IncrementalFeatureCombiner(scores=None, cutoff=5)
    meta_gs = np.linspace(2, 15, 14)
    n_cores = -1
    out_dir = 'teststack3'

    def run_parallel(sub_dir, mask_dir, test_folds, cv_folds, meta_fs, meta_gs, out_dir):

        print('Processing %s' % op.basename(sub_dir))
        mvp = DataHandler(identifier='merged').load_separate_sub(sub_dir)
        cv = StratifiedKFold(mvp.y, n_folds=test_folds)
        resultsdir = op.join(op.dirname(sub_dir), 'test_stack2')
        results = MvpResults(mvp, iterations=len(cv), resultsdir=out_dir,
                             method='averaging')

        rsc = RoiStackingClassifier(mvp, mask_type=mask_dir, folds=cv_folds,
                                    proba=True, meta_fs=meta_fs, meta_gs=meta_gs,
                                    n_cores=n_cores)

        for train_idx, test_idx in cv:
            X_train, y_train = mvp.X[train_idx, :], mvp.y[train_idx]
            X_test, y_test = mvp.X[test_idx, :], mvp.y[test_idx]
            rsc.fit(X_train, y_train)
            pred = rsc.predict(X_test, y_test)
            pred = np.argmax(pred, axis=1)
            results.update_results(test_idx=test_idx, y_pred=pred, pipeline=None)

        results.compute_and_write(directory=resultsdir)

    Parallel(n_jobs=5, verbose=5)(
        delayed(run_parallel)(sub, mask_dir, test_folds, cv_folds, meta_fs, meta_gs, out_dir) for sub in sub_dirs)

    avresults = MvpAverageResults(op.join(op.dirname(sub_dirs[0]), out_dir))
    avresults.average()