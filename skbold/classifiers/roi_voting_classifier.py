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
from skbold.data.ROIs import harvard_oxford as roi
from skbold.transformers import RoiIndexer, MeanEuclidean
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from copy import copy, deepcopy


class RoiVotingClassifier(BaseEstimator, ClassifierMixin):
    """ Voting classifier for an ensemble of patterns from different ROIs.

    This classifier fits a base-estimator (by default a linear SVM) on different
    feature sets (i.e. voxels) from different regions of interest (which are
    drawn from the Harvard-Oxford Cortical atlas), and subsequently the final
    prediction is derived through a max-voting rule, which can be either
    'soft' (argmax of mean class probability) or 'hard' (max of class
    prediction).
    """

    def __init__(self, mvp, preproc_pipeline=None, clf=None,
                 mask_type='unilateral', voting='soft', weights=None):
        """ Initializes RoiVotingClassifier object.

        Parameters
        ----------
        mvp : mvp-object
            An custom object from the skbold package containing data (X, y)
            and corresponding meta-data (e.g. mask info)
        preproc_pipeline : object
            A scikit-learn Pipeline object with desired preprocessing steps
            (e.g. scaling, additional feature selection)
        clf : object
            A scikit-learn style classifier (implementing fit(), predict(),
            and predict_proba()), that is able to be used in Pipelines.
        mask_type : str
            Can be 'unilateral' or 'bilateral', which will use all masks from
            the corresponding Harvard-Oxford Cortical (lateralized) atlas.
            Alternatively, it may be an absolute path to a directory containing
            a custom set of masks as nifti-files (default: 'unilateral').
        voting : str
            Either 'hard' or 'soft' (default: 'soft').
        weights : list (or ndarray)
            List/array of shape [n_rois] with a relative weighting factor to be
            used in the voting procedure.
        """

        self.mvp = mvp
        self.voting = voting
        self.mask_type = mask_type

        if clf is None:
            clf = SVC(C=1.0, kernel='linear', probability=True,
                      decision_function_shape='ovo')
        self.clf = clf

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

        self.pipes = [] # This will gather all roi-specific pipelines
        self.clf = clf # base-classifier

        # If no weights are specified, use equal weights
        if weights is None:
            weights = np.ones(len(self.masks))

        self.weights = weights

    def fit(self, X=None, y=None):
        """ Fits RoiVotingTransformer.

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

        for i, mask in enumerate(self.masks):
            roiindexer = RoiIndexer(self.mvp, mask, mask_threshold=0)
            pipeline = [('roiindexer', roiindexer)]
            tmp_preproc = deepcopy(self.preproc_pipeline)
            pipeline.extend(tmp_preproc.steps)
            pipeline.extend([('clf', copy(self.clf))])
            pipeline = Pipeline(pipeline)

            pipeline.fit(X, y)
            self.pipes.append(pipeline)

        return self

    def predict(self, X):
        """ Predicts class labels for X.

        Parameters
        ----------
        X : ndarray
            Array of shape = [n_samples, n_features].

        Returns
        -------
        maxvotes : ndarray
            Array with class predictions for all classes of X.
        """

        if self.voting == 'hard':
            votes = np.asarray([p.predict(X) for p in self.pipes])

            # Credits to Sebastian Rashka:
            # http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
            maxvotes = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                                           weights=self.weights)), axis=0,
                                           arr=votes)

        elif self.voting == 'soft':
            votes = np.asarray([p.predict_proba(X) for p in self.pipes])
            maxvotes = np.average(votes, axis=0, weights=self.weights).argmax(axis=-1)

        return(maxvotes)


if __name__ == '__main__':
    from skbold.utils import DataHandler
    from sklearn.cross_validation import KFold, cross_val_score

    sub_dir = '/media/lukas/data/DecodingEmotions/Validation_set/glm_zinnen/sub002'
    mvp = DataHandler(identifier='merged').load_separate_sub(sub_dir)
    mask_dir = '/home/lukas/bestrois'

    votingclassifier = RoiVotingClassifier(mvp, voting='soft',
                                           mask_type=mask_dir)
    folds = KFold(n=len(mvp.y), n_folds=20)
    scores = cross_val_score(votingclassifier, mvp.X, mvp.y, cv=folds, n_jobs=1, verbose=3)
    print(scores.mean())
