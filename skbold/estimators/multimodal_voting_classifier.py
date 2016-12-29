from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class MultimodalVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    This classifier fits a base-estimator (by default a linear SVM) on
    different feature sets of different modalities (i.e. VBM, TBSS, BOLD, etc),
    and subsequently the final prediction is derived through a max-voting rule,
    which can be either 'soft' (argmax of mean class probability) or 'hard'
    (max of class prediction).

    Notes
    -----
    This classifier has not been tested!

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
    voting : str
        Either 'hard' or 'soft' (default: 'soft').
    weights : list (or ndarray)
        List/array of shape [n_rois] with a relative weighting factor to be
        used in the voting procedure.
    """

    def __init__(self, mvp, clf=None, voting='soft', weights=None):

        self.voting = voting
        self.mvp = mvp

        if clf is None:
            clf = SVC(C=1.0, kernel='linear', probability=True,
                      decision_function_shape='ovo')
        self.clf = clf

        # If no weights are specified, use equal weights
        if weights is None:
            weights = np.ones(len(mvp.data_name))

        self.weights = weights

    def fit(self, X=None, y=None, iterations=1):
        """ Fits RoiVotingClassifier.

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

        # DOES NOTHING

        # for i, fs_id in enumerate(np.unique(self.mvp_train.featureset_id)):
        #
        #     featuresel = SelectFeatureset(self.mvp_train, fs_id)
        #     mvp_tmp = featuresel.fit().transform()
        #
        #     X = mvp_tmp.X
        #     y = mvp_tmp.y
        #
        #     pipeline = []
        #     tmp_preproc = deepcopy(self.preproc_pipeline)
        #     pipeline.extend(tmp_preproc.steps)
        #     pipeline.extend([('clf', copy(self.clf))])
        #     pipeline = Pipeline(pipeline)
        #
        #     pipeline.fit(X, y)
        #     self.pipes.append(pipeline)

        return self

    def predict(self, X):
        """ Predict class given fitted RoiVotingClassifier.

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
            maxvotes = np.average(X.round(), axis=1, weights=self.weights)

        elif self.voting == 'soft':
            maxvotes = np.average(X, axis=1, weights=self.weights).round()

        maxvotes = maxvotes.astype(int)

        return maxvotes
