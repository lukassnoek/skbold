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
from skbold.data.rois import harvard_oxford as roi
from skbold.transformers import RoiIndexer, MeanEuclidean, IncrementalFeatureCombiner
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from copy import copy, deepcopy
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV


class RoiStackingClassifier(BaseEstimator, ClassifierMixin):
    """ Stacking classifier for an ensemble of patterns from different ROIs.
    """

    def __init__(self, mvp, preproc_pipeline=None, base_clf=None, meta_clf=None,
                 mask_type='unilateral', proba=False, folds=-1, cutoff=None):
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
        self.gs_pipe = None

        if base_clf is None:
            base_clf = SVC(C=1.0, kernel='linear', probability=True,
                           decision_function_shape='ovo')
        self.base_clf = base_clf

        if meta_clf is None:
            meta_clf = LogisticRegression(multi_class='multinomial',
                                          C=0.1, solver='lbfgs')
        self.meta_clf = meta_clf

        # If no preprocessing pipeline is defined, we'll assume that at least
        # scaling and minor (univariate) feature selection is desired.
        if preproc_pipeline is None:
            scaler = StandardScaler()
            transformer = MeanEuclidean(cutoff=1, normalize=False)
            preproc_pipeline = Pipeline([('transformer', transformer),
                                         ('scaler', scaler)])

        self.preproc_pipeline = preproc_pipeline

        # Glob masks
        if mask_type not in ['unilateral', 'bilateral']:
            self.masks = glob.glob(op.join(mask_type, '*nii.gz'))
        else:
            mask_dir = op.join(op.dirname(roi.__file__), mask_type)
            self.masks = glob.glob(op.join(mask_dir, '*nii.gz'))

        if not self.masks:
            raise ValueError('No masks found in specified directory!')

        self.pipes = [] # This will gather all roi-specific pipelines
        self.folds = folds

        if cutoff is None:
            # cutoff is chance-level + 33%
            chance = (1.0 / mvp.n_class)
            cutoff = 1.333 * chance

        self.cutoff = cutoff

        # I guess it's nicer to insert RoiIndexer into the preprocessing
        # pipeline, but this crashed for some reason, so this is a temp. fix.
        self.indices = np.zeros((self.mvp.X.shape[1], len(self.masks)))
        for i, mask in enumerate(self.masks):
            roiindexer = RoiIndexer(self.mvp, mask, mask_threshold=0)
            roiindexer.fit()
            self.indices[:, i] = roiindexer.idx_
        self.indices = self.indices.astype(bool)

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

        if self.proba:
            stck_fts = np.zeros((X.shape[0], len(self.masks), self.n_class))
        else:
            stck_fts = np.zeros((X.shape[0], len(self.masks)))

        scores = np.zeros((len(self.cv), len(self.masks), self.n_class))
        self.pipes = []

        for i, fold in enumerate(self.cv):
            train_idx, test_idx = fold
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            cv_pipe = []
            for ii, mask in enumerate(self.masks):
                X_trainroi = X_train[:, self.indices[:, ii]]
                X_testroi = X_test[:, self.indices[:, ii]]
                pipeline = deepcopy(self.preproc_pipeline).steps
                pipeline.extend([('base_clf', copy(self.base_clf))])
                pipeline = Pipeline(pipeline)
                pipeline.fit(X_trainroi, y_train)

                if self.proba:
                    stck_fts[test_idx, ii, :] = pipeline.predict_proba(X_testroi)
                    scores[i, ii, :] = recall_score(y_test, pipeline.predict(X_testroi), average=None)
                else:
                    stck_fts[test_idx, ii] = pipeline.predict(X_testroi)
                    scores[i, ii, :] = recall_score(y_test, stck_fts[test_idx, ii], average=None)

                cv_pipe.append(pipeline)

            self.pipes.append(cv_pipe)

        mean_scores = scores.mean(axis=0)

        """
        if isinstance(self.cutoff, int) and self.cutoff >= 1:

            if mean_scores.ndim > 1:
                mean_scores = mean_scores.mean(axis=-1)

            best_rois = np.argsort(mean_scores)[::-1][0:self.cutoff]
            best_roi_idx = np.zeros(mean_scores.size, dtype=bool)
            best_roi_idx[best_rois] = True

        else:
            best_roi_idx = mean_scores > self.cutoff
        if self.proba:
            stck_fts = stck_fts[:, best_roi_idx]
            stck_fts = stck_fts.reshape((stck_fts.shape[0], np.prod(stck_fts.shape[1:])))
        else:
            stck_fts = stck_fts[:, best_roi_idx]
        """
        selector = IncrementalFeatureCombiner(mean_scores, cutoff=5)
        clf = deepcopy(self.meta_clf)
        gs_pipe = Pipeline([('selector', selector), ('clf', clf)])
        self.gs_pipe = GridSearchCV(gs_pipe, dict(selector__cutoff=np.linspace(0.33, .5, 2)))

        stck_tmp = stck_fts.reshape((stck_fts.shape[0], np.prod(stck_fts.shape[1:])))
        #stck_fts = selector.fit(stck_tmp).transform(stck_tmp)
        self.gs_pipe.fit(stck_tmp, y)
        self.stck_train = stck_fts
        self.train_scores = scores
        self.best_roi_idx = selector.idx_
        # meta_pred = self.meta_clf.predict(stck_fts)
        #print('Score train meta_clf: %f' % np.mean(meta_pred == y))

        return self

    def predict(self, X, y=None):

        if self.proba:
            stck_fts = np.zeros((X.shape[0], len(self.masks), self.n_class, len(self.cv)))
        else:
            stck_fts = np.zeros((X.shape[0], len(self.masks), len(self.cv)))

        scores = np.zeros((len(self.cv), len(self.masks), self.n_class))
        for i, cvfold in enumerate(self.pipes):

            for ii, pipe in enumerate(cvfold):
                Xroi = X[:, self.indices[:, ii]]
                if self.proba:
                    stck_fts[:, ii, :, i] = pipe.predict_proba(Xroi)
                else:
                    stck_fts[:, ii, i] = pipe.predict(Xroi)

                scores[i, ii, :] = recall_score(y, pipe.predict(Xroi))

        self.test_scores = scores

        if self.proba:
            stck_fts = stck_fts.mean(axis=-1)
        else:
            stck_fts = stck_fts.max(axis=-1)

        if self.proba:
            #stck_fts = stck_fts[:, self.best_roi_idx]
            stck_fts = stck_fts.reshape((stck_fts.shape[0], np.prod(stck_fts.shape[1:])))
        #else:
        #    stck_fts = stck_fts[:, self.best_roi_idx]
        meta_pred = self.gs_pipe.predict(stck_fts)
        self.stck_test = stck_fts

        return meta_pred

    def score(self, X, y):
        meta_pred = self.predict(X)
        score = np.mean(meta_pred == y)
        print('Score test meta_clf: %f' % np.mean(meta_pred == y))
        return score


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
                                                   folds=-1, proba=True, cutoff=0.4)

        for train_idx, test_idx in cv:
            X_train, y_train = mvp.X[train_idx, :], mvp.y[train_idx]
            X_test, y_test = mvp.X[test_idx, :], mvp.y[test_idx]
            stackingclassifier.fit(X_train, y_train)
            pred = stackingclassifier.predict(X_test, y_test)
            results.update_results(test_idx=test_idx, y_pred=pred, pipeline=None)

        results.compute_and_write(directory=resultsdir)
        #pred = cross_val_predict(stackingclassifier, mvp.X, mvp.y, cv=cv)

    Parallel(n_jobs=1, verbose=5)(delayed(run_parallel)(sub, mask_dir, folds=12) for sub in sub_dirs)
    avresults = MvpAverageResults(op.join(op.dirname(sub_dirs[0]), 'test_stack2'))
    avresults.average()
