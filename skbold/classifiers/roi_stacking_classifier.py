# Class to implement a stacking classifier that trains a meta-classifier
# on classifiers trained on different feature sets from different ROIs.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

# Class to implement a voting classifier from the output of
# classifiers trained on different feature sets from different ROIs.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

# Note: this implementation was inspired by the code of S. Rashka
# (http://sebastianraschka.com/Articles/2014_ensemble_classifier.html)

import glob
import os
import numpy as np
import os.path as op
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from skbold.data.rois import harvard_oxford as roi
from skbold.transformers import RoiIndexer, MeanEuclidean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from copy import copy, deepcopy


class RoiStackingClassifier(BaseEstimator, ClassifierMixin):
    """ Stacking classifier for an ensemble of patterns from different ROIs.
    """

    def __init__(self, mvp, preproc_pipeline=None, base_clf=None, meta_clf=None,
                 mask_type='unilateral', proba=False, folds=-1):
        """ Initializes RoiVotingClassifier object.

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

        if base_clf is None:
            base_clf = SVC(C=1.0, kernel='linear', probability=True,
                           decision_function_shape='ovo')
        self.base_clf = base_clf

        if meta_clf is None:
            meta_clf = SVC(C=1.0, kernel='linear', probability=True,
                           decision_function_shape='ovo')
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
            stck_fts = np.zeros((X.shape[0], self.n_class, len(self.masks)))
        else:
            stck_fts = np.zeros((X.shape[0], len(self.masks)))

        scores = np.zeros((len(self.masks), len(self.cv)))
        self.pipes = []

        for i, fold in enumerate(self.cv):
            #print('fold %i' % (i + 1))
            train_idx, test_idx = fold
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            cv_pipe = []
            for ii, mask in enumerate(self.masks):
                roiindexer = RoiIndexer(self.mvp, mask, mask_threshold=0)
                pipeline = [('roiindexer', roiindexer)]
                tmp_preproc = deepcopy(self.preproc_pipeline)
                pipeline.extend(tmp_preproc.steps)
                pipeline.extend([('base_clf', copy(self.base_clf))])
                pipeline = Pipeline(pipeline)
                pipeline.fit(X_train, y_train)

                if self.proba:
                    stck_fts[test_idx, :, ii] = pipeline.predict_proba(X_test)
                else:
                    stck_fts[test_idx, ii] = pipeline.predict(X_test)

                scores[ii, i] = pipeline.score(X_test, y_test)
                cv_pipe.append(pipeline)

            self.pipes.append(cv_pipe)

        if self.proba:
            stck_fts = stck_fts.reshape((stck_fts.shape[0], np.prod(stck_fts.shape[1:])))

        self.meta_clf.fit(stck_fts, y)
        # meta_pred = self.meta_clf.predict(stck_fts)
        #print('Score train meta_clf: %f' % np.mean(meta_pred == y))

        return self

    def predict(self, X):

        if self.proba:
            stck_fts = np.zeros((X.shape[0], self.n_class, len(self.masks), len(self.cv)))
        else:
            stck_fts = np.zeros((X.shape[0], len(self.masks), len(self.cv)))

        for i, cvfold in enumerate(self.pipes):

            for ii, pipe in enumerate(cvfold):

                if self.proba:
                    stck_fts[:, :, ii, i] = pipe.predict_proba(X)
                else:
                    stck_fts[:, ii, i] = pipe.predict(X)

        stck_fts = stck_fts.mean(axis=-1)
        if self.proba:
            stck_fts = stck_fts.reshape((stck_fts.shape[0], np.prod(stck_fts.shape[1:])))

        meta_pred = self.meta_clf.predict(stck_fts)
        return meta_pred

    def score(self, X, y):
        meta_pred = self.predict(X)
        score = np.mean(meta_pred == y)
        print('Score test meta_clf: %f' % np.mean(meta_pred == y))
        return score

if __name__ == '__main__':

    from skbold.utils import DataHandler, MvpResults, MvpAverageResults
    from joblib import Parallel, delayed

    mask_dir = '/home/lukas/bestrois'
    sub_dirs = glob.glob('/media/lukas/data/DecodingEmotions/Validation_set/glm_zinnen/sub*')

    def run_parallel(sub_dir, mask_dir=mask_dir):
        print('Processing %s' % op.basename(sub_dir))
        mvp = DataHandler(identifier='merged').load_separate_sub(sub_dir)
        cv = StratifiedKFold(mvp.y, n_folds=15)
        resultsdir = op.join(op.dirname(sub_dir), 'test_stack')
        results = MvpResults(mvp, iterations=len(cv), resultsdir=resultsdir,
                             method='averaging')
        stackingclassifier = RoiStackingClassifier(mvp, mask_type=mask_dir,
                                                   folds=-1, proba=True)

        for train_idx, test_idx in cv:
            X_train, y_train = mvp.X[train_idx, :], mvp.y[train_idx]
            X_test, y_test = mvp.X[test_idx, :], mvp.y[test_idx]
            stackingclassifier.fit(X_train, y_train)
            pred = stackingclassifier.predict(X_test)
            results.update_results(test_idx=test_idx, y_pred=pred, pipeline=None)

        results.compute_and_write(directory=resultsdir)
        #pred = cross_val_predict(stackingclassifier, mvp.X, mvp.y, cv=cv)

    Parallel(n_jobs=-1, verbose=5)(delayed(run_parallel)(sub, mask_dir) for sub in sub_dirs)
    avresults = MvpAverageResults(op.join(op.dirname(sub_dirs[0]), 'test_stack'))
    avresults.average()
    # To do: nonlinear metaclf (rbf)