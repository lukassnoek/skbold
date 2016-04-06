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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from skbold.data.ROIs import harvard_oxford as roi
from skbold.transformers import RoiIndexer, MeanEuclidean, IncrementalFeatureCombiner
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from copy import copy, deepcopy
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings('ignore')  # hack to turn off UndefinedMetricWarning


class RoiStackingClassifier(BaseEstimator, ClassifierMixin):
    """ Stacking classifier for an ensemble of patterns from different ROIs.
    """

    def __init__(self, mvp, preproc_pipe='default', base_clf=None, meta_clf=None,
                 mask_type='unilateral', proba=False, folds=-1, meta_fs='univar',
                 meta_gs=None):
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

        """ STEP 1: PREPROCESSING """
        # If no preprocessing pipeline is defined, we'll assume that at least
        # scaling and minor (univariate) feature selection is desired.
        if preproc_pipe == 'default':
            scaler = StandardScaler()
            transformer = MeanEuclidean(cutoff=1, normalize=False)
            preproc_pipe = Pipeline([('transformer', transformer),
                                     ('scaler', scaler)])
        self.preproc_pipe = preproc_pipe

        """ STEP 2a: BASE CLASSIFIER """
        if base_clf is None:
            base_clf = SVC(C=1.0, kernel='linear', probability=True,
                           decision_function_shape='ovo')
        self.base_clf = base_clf

        """ STEP 2b: PREPROC+BASE """
        base_pipe = self.preproc_pipe.steps
        base_pipe.extend([('base_clf', self.base_clf)])
        self.base_pipe = Pipeline(base_pipe)

        """ STEP 3a: META FEATURE SELECTION + CLASSIFICATION """
        if meta_clf is None:
            meta_clf = LogisticRegression(multi_class='multinomial',
                                          C=0.1, solver='lbfgs')
        self.meta_fs = meta_fs
        meta_pipe = Pipeline([('selector', meta_fs),
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
            mask_dir = op.join(op.dirname(roi.__file__), mask_type)
            self.masks = glob.glob(op.join(mask_dir, '*nii.gz'))

        if not self.masks:
            raise ValueError('No masks found in specified directory!')

        self.folds = folds

        # I guess it's nicer to insert RoiIndexer into the preprocessing
        # pipeline, but this crashed for some reason, so this is a temp. fix.
        self.indices = np.zeros((self.mvp.X.shape[1], len(self.masks)))
        for i, mask in enumerate(self.masks):
            roiindexer = RoiIndexer(self.mvp, mask, mask_threshold=0)
            roiindexer.fit()
            self.indices[:, i] = roiindexer.idx_
        self.indices = self.indices.astype(bool)

        # Metrics
        self.train_roi_scores = None
        self.test_roi_scores = None

    def fit(self, X=None, y=None):
        """ Fits RoiStackingTransformer.

        Parameters
        ----------
        X : ndarray
            Array of shape = [n_samples, n_features]. If None (default), X
            is drawn from the mvp object.
        y : List[str] or numpy ndarray[str]
            List or ndarray with floats corresponding to labels.
        """

        if X is None:
            X = self.mvp.X

        if y is None:
            y = self.mvp.y

        if self.folds == -1:
            self.folds = (y == 0).sum()
        self.cv = StratifiedKFold(y, n_folds=self.folds)

        n_trials = X.shape[0]
        n_class = self.n_class
        n_masks = len(self.masks)
        n_folds = len(self.cv)

        if self.proba:  # trials X masks X nr. classes (proba output)
            stck_fts = np.zeros((n_trials, n_masks, n_class))
        else:  # discrete class output
            stck_fts = np.zeros((n_trials, n_masks))

        scores = np.zeros((len(self.cv), len(self.masks), self.n_class))
        self.base_pipes = []  # This will gather all roi-specific pipelines

        # Loop over inner cv-folds to generate post. probs. for all trials
        for i, fold in enumerate(self.cv):
            train_idx, test_idx = fold
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            roi_pipes = []
            for ii, mask in enumerate(self.masks):
                X_trainroi = X_train[:, self.indices[:, ii]]
                X_testroi = X_test[:, self.indices[:, ii]]
                base_pipe = deepcopy(self.base_pipe)
                base_pipe.fit(X_trainroi, y_train)

                # Save precision to use in selection of meta-features
                pred = base_pipe.predict(X_testroi)
                scores[i, ii, :] = precision_score(y_test, pred, average=None)

                if self.proba:  # save posterior probabilities
                    stck_fts[test_idx, ii, :] = base_pipe.predict_proba(X_testroi)
                else:  # save discrete class predictions
                    stck_fts[test_idx, ii] = pred

                roi_pipes.append(base_pipe)

            self.base_pipes.append(roi_pipes)

        mean_scores = scores.mean(axis=0)  # average over cv folds
        stck_tmp = stck_fts.reshape((stck_fts.shape[0], np.prod(stck_fts.shape[1:])))

        #meta_clf = deepcopy(self.meta_pipe)
        if self.meta_fs.__class__ == IncrementalFeatureCombiner:
            if self.is_gridsearch(self.meta_pipe):
                self.meta_pipe.estimator.set_params(selector__scores=mean_scores)
            else:
                self.meta_pipe.set_params(selector__scores=mean_scores)

        self.meta_pipe.fit(stck_tmp, y)

        # Save some stuff for exploratory stuff / debugging
        self.stck_train = stck_tmp
        self.train_roi_scores = mean_scores

        if self.is_gridsearch(self.meta_pipe):
            self.best_roi_idx = self.meta_pipe.best_estimator_.named_steps['selector'].idx_
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
        n_folds = len(self.cv)

        if self.proba:
            stck_fts = np.zeros((n_trials, n_masks, n_class, n_folds))
        else:
            stck_fts = np.zeros((n_trials, n_masks, n_folds))

        scores = np.zeros((n_folds, n_masks, n_class))
        for i, cvfold in enumerate(self.base_pipes):

            for ii, pipe in enumerate(cvfold):
                Xroi = X[:, self.indices[:, ii]]

                pred = pipe.predict(Xroi)
                scores[i, ii, :] = precision_score(y, pred, average=None)

                if self.proba:
                    stck_fts[:, ii, :, i] = pipe.predict_proba(Xroi)
                else:
                    stck_fts[:, ii, i] = pred

        if self.proba:
            stck_fts = stck_fts.mean(axis=-1)
        else:
            stck_fts = stck_fts.max(axis=-1)

        if self.proba:
            shape = stck_fts.shape
            stck_fts = stck_fts.reshape((shape[0], np.prod(shape[1:])))

        meta_pred = self.meta_pipe.predict(stck_fts)
        self.stck_test = stck_fts
        self.test_roi_scores = scores.mean(axis=0)

        return meta_pred

    def is_gridsearch(self, pipe):

        if hasattr(pipe, 'estimator'):
            return True
        else:
            return False

if __name__ == '__main__':

    from skbold.utils import DataHandler, MvpResults, MvpAverageResults
    from joblib import Parallel, delayed

    mask_dir = 'unilateral'
    sub_dirs = glob.glob('/media/lukas/data/DecodingEmotions/Validation_set/glm_zinnen/sub*')

    def run_parallel(sub_dir, mask_dir=mask_dir, folds=10):
        print('Processing %s' % op.basename(sub_dir))
        mvp = DataHandler(identifier='merged').load_separate_sub(sub_dir)
        cv = StratifiedKFold(mvp.y, n_folds=folds)
        resultsdir = op.join(op.dirname(sub_dir), 'test_stack2')
        results = MvpResults(mvp, iterations=len(cv), resultsdir=resultsdir,
                             method='averaging')
        stackingclassifier = RoiStackingClassifier(mvp, mask_type=mask_dir,
                                                   folds=-1, proba=True,
                                                   meta_fs=IncrementalFeatureCombiner(scores=None, cutoff=0.33),
                                                   meta_gs=np.linspace(0.3, 0.5, 10))

        for train_idx, test_idx in cv:
            X_train, y_train = mvp.X[train_idx, :], mvp.y[train_idx]
            X_test, y_test = mvp.X[test_idx, :], mvp.y[test_idx]
            stackingclassifier.fit(X_train, y_train)
            pred = stackingclassifier.predict(X_test, y_test)
            results.update_results(test_idx=test_idx, y_pred=pred, pipeline=None)

        results.compute_and_write(directory=resultsdir)

    Parallel(n_jobs=1, verbose=5)(delayed(run_parallel)(sub, mask_dir, folds=12) for sub in sub_dirs)
    avresults = MvpAverageResults(op.join(op.dirname(sub_dirs[0]), 'test_stack2'))
    avresults.average()